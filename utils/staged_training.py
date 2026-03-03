"""
Staged training scheduler with adaptive monitoring.
Manages multi-stage loss weight transitions and reconstruction quality monitoring.

This module implements a 3-stage training approach for VAE-GMM:
- Stage 1: VAE warmup (GMM disabled, minimal KL pressure)
- Stage 2: Gradual transition (linear interpolation to target weights)
- Stage 3: Adaptive monitoring (automatic rebalancing based on reconstruction quality)

Based on findings from Round 3 standalone VAE tuning:
- Critical reconstruction loss threshold: 5470 (separates reasonable from collapsed quality)
- Optimal KL weight: 0.1 (lower weight gives better reconstruction AND higher KL divergence)
- Optimal learning rate: 0.0001 (slower learning prevents collapse)
"""

from typing import Dict


class StagedTrainingScheduler:
    """
    Manages loss weight scheduling across training stages.

    Supports:
    - Multi-stage training (VAE warmup → gradual transition → adaptive monitoring)
    - Linear interpolation between stages
    - Adaptive weight adjustment based on metrics
    - Stage transition logging

    Attributes:
        config: Configuration dictionary with stage definitions
        current_stage: Current training stage (1, 2, or 3)
        best_recon_loss: Best reconstruction loss seen so far (for adaptive monitoring)
        recon_history: History of reconstruction losses for tracking trends
    """

    def __init__(self, config: Dict):
        """
        Initialize staged training scheduler from config.

        Args:
            config: Dictionary with staged_training configuration containing:
                - stage1_end_epoch: When Stage 1 ends
                - stage2_end_epoch: When Stage 2 ends
                - stage1_weights: Loss weights for Stage 1 {reconstruction, kl, gmm}
                - stage2_weights: Target loss weights for Stage 2/3
                - adaptive_monitoring: Adaptive monitoring configuration
        """
        self.config = config
        self.current_stage = 1
        self.best_recon_loss = float('inf')
        self.recon_history = []

    def get_weights_for_epoch(self, epoch: int, current_metrics: Dict) -> Dict:
        """
        Compute loss weights for current epoch based on stage.

        Args:
            epoch: Current training epoch (1-indexed)
            current_metrics: Dictionary with current metrics (recon_loss, kl_loss, etc.)

        Returns:
            Dictionary with loss weights: {'reconstruction': float, 'kl': float, 'gmm': float}
        """
        # Stage 1: VAE Warmup
        if epoch <= self.config['stage1_end_epoch']:
            self.current_stage = 1
            return self._stage1_weights()

        # Stage 2: Gradual Transition
        elif epoch <= self.config['stage2_end_epoch']:
            self.current_stage = 2
            return self._stage2_weights(epoch)

        # Stage 3: Adaptive Monitoring
        else:
            self.current_stage = 3
            return self._stage3_weights(epoch, current_metrics)

    def _stage1_weights(self) -> Dict:
        """
        Stage 1: VAE warmup.

        Disable GMM and use minimal KL pressure to ensure the VAE
        learns good reconstruction before adding clustering objectives.

        Returns:
            Dictionary with Stage 1 loss weights
        """
        return self.config['stage1_weights'].copy()

    def _stage2_weights(self, epoch: int) -> Dict:
        """
        Stage 2: Gradual transition.

        Linearly interpolate from Stage 1 weights to Stage 2 target weights.
        This smooth transition prevents sudden loss landscape changes that
        can cause training instability.

        Args:
            epoch: Current epoch (used to compute interpolation progress)

        Returns:
            Dictionary with interpolated loss weights
        """
        start_epoch = self.config['stage1_end_epoch']
        end_epoch = self.config['stage2_end_epoch']
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)

        # Interpolate between stage1 and stage2 target weights
        weights = {}
        for key in ['reconstruction', 'kl', 'gmm']:
            start_val = self.config['stage1_weights'][key]
            end_val = self.config['stage2_weights'][key]
            weights[key] = start_val + (end_val - start_val) * progress

        return weights

    def _stage3_weights(self, epoch: int, metrics: Dict) -> Dict:
        """
        Stage 3: Adaptive monitoring.

        Use target weights from Stage 2, but monitor reconstruction quality
        and automatically rebalance if significant degradation is detected.

        This adaptive mechanism prevents the catastrophic collapse seen in
        previous joint VAE-GMM training attempts (Round 2: recon 6400-7100).

        Args:
            epoch: Current epoch
            metrics: Current metrics dictionary (must contain 'recon_loss')

        Returns:
            Dictionary with loss weights (possibly adjusted for rebalancing)
        """
        weights = self.config['stage2_weights'].copy()

        # Check for reconstruction degradation
        if self.config['adaptive_monitoring']['enabled']:
            recon_loss = metrics.get('recon_loss', float('inf'))
            self.recon_history.append(recon_loss)

            # Update best reconstruction loss seen so far
            if recon_loss < self.best_recon_loss:
                self.best_recon_loss = recon_loss

            # Check degradation every N epochs
            check_interval = self.config['adaptive_monitoring']['check_interval']
            if epoch % check_interval == 0:
                recon_delta = recon_loss - self.best_recon_loss
                degradation_tolerance = self.config['adaptive_monitoring']['recon_degradation_tolerance']

                # Significant degradation detected
                if recon_delta > degradation_tolerance:
                    rebalance_factor = self.config['adaptive_monitoring']['rebalance_factor']
                    weights['reconstruction'] *= rebalance_factor
                    weights['kl'] /= rebalance_factor
                    weights['gmm'] /= rebalance_factor
                    print(f"\n⚠️  ADAPTIVE REBALANCING at epoch {epoch}:")
                    print(f"   Reconstruction degraded: +{recon_delta:.0f}")
                    print(f"   New weights: recon={weights['reconstruction']:.2f}, "
                          f"kl={weights['kl']:.3f}, gmm={weights['gmm']:.2f}")

                # Critical threshold exceeded (from Round 3 discovery: 5470)
                critical_threshold = self.config['adaptive_monitoring']['recon_critical_threshold']
                if recon_loss > critical_threshold:
                    print(f"\n🔴 CRITICAL THRESHOLD EXCEEDED at epoch {epoch}:")
                    print(f"   Reconstruction loss {recon_loss:.0f} > {critical_threshold}")
                    print(f"   Emergency rebalancing: Doubling reconstruction weight")
                    weights['reconstruction'] *= 2.0
                    weights['kl'] *= 0.5
                    weights['gmm'] *= 0.5

        return weights

    def log_stage_transition(self, epoch: int, logger) -> None:
        """
        Log stage transitions for visibility.

        Logs clear messages when transitioning between stages to help
        understand the training dynamics.

        Args:
            epoch: Current epoch
            logger: Logger instance with log_message() method
        """
        if epoch == 1:
            logger.log_message(f"\n{'='*80}")
            logger.log_message(f"STAGED TRAINING - STAGE 1: VAE WARMUP (Epochs 1-{self.config['stage1_end_epoch']})")
            logger.log_message(f"Goal: Achieve reconstruction loss < 5000")
            logger.log_message(f"Weights: {self.config['stage1_weights']}")
            logger.log_message(f"{'='*80}\n")

        elif epoch == self.config['stage1_end_epoch'] + 1:
            logger.log_message(f"\n{'='*80}")
            logger.log_message(f"STAGED TRAINING - STAGE 2: GRADUAL TRANSITION (Epochs {epoch}-{self.config['stage2_end_epoch']})")
            logger.log_message(f"Goal: Smoothly shift to balanced VAE-GMM training")
            logger.log_message(f"Target weights: {self.config['stage2_weights']}")
            logger.log_message(f"{'='*80}\n")

        elif epoch == self.config['stage2_end_epoch'] + 1:
            logger.log_message(f"\n{'='*80}")
            logger.log_message(f"STAGED TRAINING - STAGE 3: ADAPTIVE MONITORING (Epochs {epoch}-end)")
            logger.log_message(f"Goal: Maintain balanced training with adaptive safety net")
            logger.log_message(f"Monitoring: Reconstruction loss (critical threshold: {self.config['adaptive_monitoring']['recon_critical_threshold']})")
            logger.log_message(f"{'='*80}\n")

    def get_stage_summary(self) -> Dict:
        """
        Get summary of current training stage.

        Returns:
            Dictionary with stage information:
                - current_stage: Current stage number (1, 2, or 3)
                - best_recon_loss: Best reconstruction loss seen
                - recon_history_length: Number of epochs tracked
        """
        return {
            'current_stage': self.current_stage,
            'best_recon_loss': self.best_recon_loss,
            'recon_history_length': len(self.recon_history)
        }
