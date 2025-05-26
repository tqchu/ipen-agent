import logging
import os.path

from openai import OpenAI
import json

from cache.cache import GeneralCache
from config.config import Config
from exploitdb.exploitdb import ExploitDbResolver

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class LLMExploitAnalyzer:
    """
    A helper class to:
      1) Analyze exploit code with an LLM and extract structured metadata (in JSON).
      2) Optionally refactor or fix the exploit code if it's broken or incomplete.

    Requires 'openai>=1.0.0'
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize the analyzer with OpenAI client and model.

        :param model: The OpenAI model name to use (e.g., 'gpt-4', 'gpt-3.5-turbo').
        """
        # Initialize OpenAI client with API key from Config
        api_key = Config().get("OPEN_AI_KEY")
        self.client = OpenAI(api_key=api_key,
                             # base_url="https://models.inference.ai.azure.com"
                             )
        self.exploit_db = ExploitDbResolver(cache_file_path="/home/truongchu/Academic/Graduation_Thesis/Project/AI/ipen-agent/cache/data/exploitdb_cache.pkl")
        self.model = model
        self.cache = GeneralCache(
            "/home/truongchu/Academic/Graduation_Thesis/Project/AI/ipen-agent/cache/data/exploitdb_cache.pkl",
            cache_func=None)

    # Update the analyze method to clean up the output files
    def analyze(self, file_name: str, metadata_outfile: str = None, code_outfile: str = None) -> (str, str):
        """
        Analyze exploit code and save results to files if specified.

        Args:
            exploit_code: The raw exploit code to analyze
            metadata_outfile: Optional path to save metadata JSON
            code_outfile: Optional path to save refactored code

        Returns:
            Tuple of (metadata_json, refactored_code)
        """
        with open(file_name, 'r') as f:
            exploit_code = f.read()

        refactored_exploit_code = self._fix_exploit_code(file_name, exploit_code)
        metadata_json = self._generate_exploit_metadata(refactored_exploit_code)

        clean_metadata = None
        if metadata_outfile:
            # clean like code_out_file
            clean_metadata = metadata_json
            if clean_metadata.startswith('```'):
                # Find the first newline after the opening backticks
                first_newline = clean_metadata.find('\n')
                if first_newline > 0:
                    clean_metadata = clean_metadata[first_newline + 1:]
            if clean_metadata.endswith('```'):
                # Find the last newline before the closing backticks
                last_newline = clean_metadata.rfind('\n', 0, clean_metadata.rfind('```'))
                if last_newline > 0:
                    clean_metadata = clean_metadata[:last_newline]

        metadata_json_t = json.loads(clean_metadata)
        metadata_json_t["exploit"] = self.exploit_db.find_exploit_by_id(file_name.split('.')[0])

        metadata_json = json.dumps(metadata_json_t)
        # Save metadata to file if path provided
        if metadata_outfile:
            # clean like code_out_file
            clean_metadata = metadata_json
            if clean_metadata.startswith('```'):
                # Find the first newline after the opening backticks
                first_newline = clean_metadata.find('\n')
                if first_newline > 0:
                    clean_metadata = clean_metadata[first_newline + 1:]
            if clean_metadata.endswith('```'):
                # Find the last newline before the closing backticks
                last_newline = clean_metadata.rfind('\n', 0, clean_metadata.rfind('```'))
                if last_newline > 0:
                    clean_metadata = clean_metadata[:last_newline]

            with open(metadata_outfile, 'w') as f:
                f.write(clean_metadata)

        # Save refactored code to file if path provided
        if code_outfile:
            # Remove any triple backticks that might be in the code
            clean_code = refactored_exploit_code
            if clean_code.startswith('```'):
                # Find the first newline after the opening backticks
                first_newline = clean_code.find('\n')
                if first_newline > 0:
                    clean_code = clean_code[first_newline + 1:]

            if clean_code.endswith('```'):
                # Find the last newline before the closing backticks
                last_newline = clean_code.rfind('\n', 0, clean_code.rfind('```'))
                if last_newline > 0:
                    clean_code = clean_code[:last_newline]

            initial_extension = code_outfile.split('.')[-1]
            if json.loads(metadata_json)["extension"] != f".{initial_extension}":
                code_outfile = f"{code_outfile.split('.')[0]}{json.loads(metadata_json)['extension']}"

            with open(code_outfile, 'w') as f:
                f.write(clean_code)

        return metadata_json, refactored_exploit_code

    def _generate_exploit_metadata(self, exploit_code: str) -> str:
        """
        Analyze the given exploit code and extract structured JSON metadata.

        :param exploit_code: The raw exploit code (as a string).
        :return: A JSON string containing the metadata.
        """
        system_prompt = (
            "You are an AI specialized in reading exploit code and outputting structured JSON metadata. "
            "Identify the code language and output its extension (must include dot sign, maybe .sh, .py or .c, or other extensions) (field name is extension),"
            "needed arguments (field name is arguments) (including two fields name and type, the arguments must be in the order that the code needs, name must in [host, port, username, password, token, command, file_path, ssl, timeout], type must in [string,int,bool], should return only needed arguments of the code;"
            "if no argument needed, return empty list), a short description, critical level of the vulnerability [high, medium, low] (field name is level)."
            "Also, identify the code is local or remote execution (field name is location_type, in [local, remote], identify the code is interactive or one time (field name is is_interactive, in [true, false]),"
            "and identify the code is Privilege Escalation or not (field name is is_privilege_escalation, in [true, false])."
            ". Estimate your confidence in parsing it (field name is confidence) (from 0 to 1, in real number). "
            "Return ONLY valid JSON. No extra commentary."
            "An example is as follows:"
            """{
  "extension": ".py",
  "arguments": [
    {
      "name": "host",
      "type": "string"
    },
    {
      "name": "port",
      "type": "int"
    }
  ],
  "description": "This Python exploit code establishes a reverse shell connection to a specified host and port, allowing remote code execution.",
  "level": "high",
  "confidence": 0.9,
  "location_type": "remote",
    "is_interactive": false,
    "is_privilege_escalation": false
}"""
        )

        user_prompt = f"Exploit Code:\n{exploit_code}\n"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2  # Lower temperature for more focused, consistent output
        )

        # The new API returns the content directly in the response
        content = response.choices[0].message.content.strip()
        return content

    def _fix_exploit_code(self, file_name: str, exploit_code: str) -> str:
        """
        Attempt to fix or refactor the given exploit code.

        :param exploit_code: The raw exploit code to be fixed/refactored.
        :return: A string with the updated (hopefully working) exploit code.
        """
        system_prompt = (
            "You are an assistant that fixes and refactors exploit code. "
            "If the code has commentary but not annotated with slash comment, remove these comments, if have two code programs in one file, choose the first one. "
            "Address syntax errors, missing libraries, and optionally convert hard-coded values "
            "into arguments. Do NOT add extra commentary or disclaimers. Return ONLY code."
            "Also, the exploit code needs arguments like [host, port, username, password, token, command, file_path, ssl, timeout], so if these arguments is hard-coded, you must convert it to the code that accepts arguments."
            "In conclusion, you should fix the code to be a single working file that can be run directly. If the code already works, return the same code."
        )

        user_prompt = f"Here is file name of the exploit code: {file_name} (if it is txt, you should detect the language) and the exploit code to fix:\n{exploit_code}\n"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2  # keep responses consistent
        )

        choices = response.choices
        if not choices or len(choices) == 0:
            return ""

        fixed_code = response.choices[0].message.content.strip()
        return fixed_code

    def generate_from_exploitdb_cache(self):
        for _, exploit_paths in self.cache.data.items():
            for exploit_path in exploit_paths:
                file_name = exploit_path.split('/')[-1]

                output_folder = f"exploits/{file_name.split('.')[0]}"

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                elif len(os.listdir(output_folder)) < 2:  # Folder exists but is empty
                    pass  # Continue processing
                else:
                    # Skip processing as folder exists and has content
                    continue

                _, _ = self.analyze(exploit_path, f"{output_folder}/metadata.json", f"{output_folder}/{file_name}")

                logging.info("Generated metadata and refactored code for exploit: %s", exploit_path)


# # Usage example
# if __name__ == "__main__":
#     analyzer = LLMExploitAnalyzer()
#
#     # Generate metadata
#     # analyzer.generate_from_exploitdb_cache()
#     # analyzer.analyze("/home/truongchu/Academic/Graduation_Thesis/Project/AI/ipen-agent/llm/exploits/8037/8037.py", "metadata_fixed.json", "8037_fixed.py")
#     analyzer.generate_from_exploitdb_cache()
