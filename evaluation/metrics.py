from torch.utils.tensorboard import SummaryWriter  # or tensorboardX, etc.


class MetricsLogger:
    """Logs training metrics (episode rewards, lengths, success flags) and provides evaluation utilities."""

    def __init__(self, log_dir="runs"):
        """
        Initialize the metrics logger.
        :param log_dir: Directory for TensorBoard logs.
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.episode_count = 0
        self.success_count = 0  # track how many episodes achieved the goal

    def log_episode(self, episode, reward, length, goal_reached):
        """
        Log metrics for a single episode.
        :param episode: Episode index.
        :param reward: Cumulative reward obtained in the episode.
        :param length: Number of steps (actions) in the episode.
        :param goal_reached: Boolean, whether the penetration goal was achieved.
        """
        # Log to TensorBoard
        self.writer.add_scalar("Episode/Reward", reward, episode)
        self.writer.add_scalar("Episode/Length", length, episode)
        self.writer.add_scalar("Episode/GoalReached", int(goal_reached), episode)
        # Update counters for success rate calculation
        self.episode_count += 1
        if goal_reached:
            self.success_count += 1
        # Also log to console for immediate feedback
        import logging
        logging.info(f"Episode {episode}: reward={reward:.2f}, length={length}, goal_reached={goal_reached}")

    def success_rate(self):
        """Compute the success rate (fraction of episodes where goal was reached)."""
        if self.episode_count == 0:
            return 0.0
        return self.success_count / self.episode_count

    def close(self):
        """Close out the logger (flushes TensorBoard data)."""
        self.writer.close()
