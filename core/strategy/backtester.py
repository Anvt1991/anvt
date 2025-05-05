#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtester for stock market prediction strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtester:
    """
    Backtester for stock market prediction strategies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the backtester
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "initial_capital": 10000000,  # Initial capital in VND
            "position_size": 0.1,  # Percentage of capital to use for each position
            "commission_rate": 0.001,  # Commission rate (0.1%)
            "slippage": 0.001,  # Slippage rate (0.1%)
            "risk_free_rate": 0.03,  # Risk-free rate (3% annual)
            "trading_days_per_year": 252,  # Number of trading days in a year
            "signal_threshold": 0.55,  # Threshold for buy signal (for ML model predictions)
            "exit_threshold": 0.45,  # Threshold for sell signal (for ML model predictions)
            "stop_loss": 0.05,  # Stop loss percentage (5%)
            "take_profit": 0.15,  # Take profit percentage (15%)
            "max_positions": 5,  # Maximum number of positions at the same time
            "rebalance_frequency": "daily"  # Rebalance frequency: 'daily', 'weekly', 'monthly'
        }
        
        # Update config with default values for missing keys
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Initialize state
        self.reset()
        
        logger.info("Backtester initialized")
    
    def reset(self):
        """Reset the backtester to initial state"""
        self.capital = self.config["initial_capital"]
        self.equity = self.config["initial_capital"]
        self.positions = {}  # Symbol -> Position info
        self.trades = []
        self.equity_curve = []
        self.signals = []
        self.performance_metrics = {}
    
    def run(self, data: Dict[str, pd.DataFrame], signals: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run backtest
        
        Args:
            data: Dictionary mapping symbol to price DataFrame (with OHLCV data)
            signals: Dictionary mapping symbol to signal DataFrame (with prediction signals)
            
        Returns:
            Dictionary with backtest results
        """
        if not data or not signals:
            logger.error("No data or signals provided for backtest")
            return {"error": "No data or signals provided"}
        
        try:
            # Reset state
            self.reset()
            
            # Get all dates from all symbols and sort
            all_dates = set()
            for symbol, df in data.items():
                if isinstance(df.index, pd.DatetimeIndex):
                    all_dates.update(df.index)
                else:
                    logger.warning(f"DataFrame for {symbol} does not have DatetimeIndex")
            
            all_dates = sorted(all_dates)
            
            if not all_dates:
                logger.error("No valid dates found in data")
                return {"error": "No valid dates found in data"}
            
            # Initialize equity curve with initial capital
            self.equity_curve = [{
                "date": all_dates[0],
                "capital": self.capital,
                "equity": self.equity,
                "positions": 0,
                "returns": 0.0
            }]
            
            # Simulate trading for each date
            for current_date in all_dates[1:]:  # Start from the second date
                # Update existing positions
                self._update_positions(current_date, data)
                
                # Process signals for this date
                self._process_signals(current_date, data, signals)
                
                # Record equity
                self._record_equity(current_date)
            
            # Calculate performance metrics
            self.performance_metrics = self._calculate_performance_metrics()
            
            # Prepare results
            results = {
                "equity_curve": pd.DataFrame(self.equity_curve),
                "trades": pd.DataFrame(self.trades) if self.trades else pd.DataFrame(),
                "signals": pd.DataFrame(self.signals) if self.signals else pd.DataFrame(),
                "performance_metrics": self.performance_metrics,
                "final_equity": self.equity,
                "total_return": (self.equity / self.config["initial_capital"] - 1) * 100
            }
            
            logger.info(f"Backtest completed - Final equity: {self.equity:,.0f} VND, "
                      f"Total return: {results['total_return']:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            return {"error": f"Backtest failed: {str(e)}"}
    
    def _update_positions(self, current_date: datetime, data: Dict[str, pd.DataFrame]):
        """
        Update existing positions
        
        Args:
            current_date: Current date
            data: Dictionary mapping symbol to price DataFrame
        """
        # List of positions to close
        positions_to_close = []
        
        # Update each position
        for symbol, position in self.positions.items():
            if symbol not in data:
                logger.warning(f"No data for {symbol} - skipping position update")
                continue
            
            # Get price data for this date
            df = data[symbol]
            if current_date not in df.index:
                # No data for this date
                continue
            
            # Get current price
            current_price = df.loc[current_date, 'close']
            
            # Calculate profit/loss
            entry_price = position['entry_price']
            quantity = position['quantity']
            position_pnl = (current_price - entry_price) * quantity
            position_pnl_pct = (current_price / entry_price - 1) * 100
            
            # Update position info
            self.positions[symbol].update({
                'current_price': current_price,
                'current_value': current_price * quantity,
                'profit_loss': position_pnl,
                'profit_loss_pct': position_pnl_pct,
                'days_held': (current_date - position['entry_date']).days
            })
            
            # Check stop loss
            if position_pnl_pct <= -self.config["stop_loss"] * 100:
                positions_to_close.append((symbol, "Stop Loss"))
            
            # Check take profit
            elif position_pnl_pct >= self.config["take_profit"] * 100:
                positions_to_close.append((symbol, "Take Profit"))
        
        # Close positions
        for symbol, reason in positions_to_close:
            self._close_position(current_date, symbol, reason)
    
    def _process_signals(self, current_date: datetime, data: Dict[str, pd.DataFrame], 
                         signals: Dict[str, pd.DataFrame]):
        """
        Process signals for the current date
        
        Args:
            current_date: Current date
            data: Dictionary mapping symbol to price DataFrame
            signals: Dictionary mapping symbol to signal DataFrame
        """
        # Check if we can open new positions (max positions limit)
        available_slots = self.config["max_positions"] - len(self.positions)
        
        if available_slots <= 0:
            return
        
        # Process signals for each symbol
        buy_candidates = []
        
        for symbol, signal_df in signals.items():
            if symbol not in data:
                logger.warning(f"No price data for {symbol} - skipping signal processing")
                continue
            
            # Skip if already in position
            if symbol in self.positions:
                continue
            
            # Get signal for this date
            if current_date not in signal_df.index:
                # No signal for this date
                continue
            
            # Get signal value and price
            signal_row = signal_df.loc[current_date]
            
            if 'probability' in signal_row:
                signal_value = signal_row['probability']
            elif 'signal' in signal_row:
                signal_value = signal_row['signal']
            else:
                # No valid signal column
                continue
            
            # Record signal
            self.signals.append({
                'date': current_date,
                'symbol': symbol,
                'signal_value': signal_value
            })
            
            # Check if signal is above threshold
            if signal_value >= self.config["signal_threshold"]:
                # Get price data for this date
                price_df = data[symbol]
                if current_date not in price_df.index:
                    # No price data for this date
                    continue
                
                # Get price and add to candidates
                current_price = price_df.loc[current_date, 'close']
                
                buy_candidates.append({
                    'symbol': symbol,
                    'signal_value': signal_value,
                    'price': current_price
                })
        
        # Sort candidates by signal strength and open positions
        if buy_candidates:
            buy_candidates.sort(key=lambda x: x['signal_value'], reverse=True)
            
            for candidate in buy_candidates[:available_slots]:
                self._open_position(current_date, candidate['symbol'], candidate['price'], candidate['signal_value'])
    
    def _open_position(self, date: datetime, symbol: str, price: float, signal_value: float):
        """
        Open a new position
        
        Args:
            date: Entry date
            symbol: Symbol
            price: Entry price
            signal_value: Signal value
        """
        # Calculate position size
        position_value = self.capital * self.config["position_size"]
        
        # Calculate quantity (shares)
        quantity = int(position_value / price)
        
        if quantity <= 0:
            logger.warning(f"Cannot open position for {symbol} - insufficient capital")
            return
        
        # Calculate actual position value
        actual_value = quantity * price
        
        # Calculate commission
        commission = actual_value * self.config["commission_rate"]
        
        # Update capital
        self.capital -= (actual_value + commission)
        
        # Record position
        self.positions[symbol] = {
            'entry_date': date,
            'entry_price': price,
            'quantity': quantity,
            'value': actual_value,
            'commission': commission,
            'signal_value': signal_value,
            'current_price': price,
            'current_value': actual_value,
            'profit_loss': 0.0,
            'profit_loss_pct': 0.0,
            'days_held': 0
        }
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'buy',
            'price': price,
            'quantity': quantity,
            'value': actual_value,
            'commission': commission,
            'signal_value': signal_value
        })
        
        logger.debug(f"Opened position for {symbol} - {quantity} shares at {price:,.0f} VND - "
                    f"Value: {actual_value:,.0f} VND - Signal: {signal_value:.2f}")
    
    def _close_position(self, date: datetime, symbol: str, reason: str = "Signal"):
        """
        Close an existing position
        
        Args:
            date: Exit date
            symbol: Symbol
            reason: Reason for closing position
        """
        if symbol not in self.positions:
            logger.warning(f"Cannot close position for {symbol} - no open position")
            return
        
        # Get position details
        position = self.positions[symbol]
        
        # Calculate exit value
        exit_price = position['current_price']
        quantity = position['quantity']
        exit_value = exit_price * quantity
        
        # Calculate commission
        commission = exit_value * self.config["commission_rate"]
        
        # Calculate profit/loss
        profit_loss = position['profit_loss'] - commission
        profit_loss_pct = (exit_price / position['entry_price'] - 1) * 100
        
        # Update capital
        self.capital += (exit_value - commission)
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'sell',
            'price': exit_price,
            'quantity': quantity,
            'value': exit_value,
            'commission': commission,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'days_held': position['days_held'],
            'reason': reason
        })
        
        logger.debug(f"Closed position for {symbol} - {quantity} shares at {exit_price:,.0f} VND - "
                    f"P/L: {profit_loss:,.0f} VND ({profit_loss_pct:.2f}%) - Reason: {reason}")
        
        # Remove position
        del self.positions[symbol]
    
    def _record_equity(self, date: datetime):
        """
        Record equity for the current date
        
        Args:
            date: Current date
        """
        # Calculate total position value
        position_value = sum(pos['current_value'] for pos in self.positions.values())
        
        # Calculate equity
        equity = self.capital + position_value
        
        # Calculate daily return
        prev_equity = self.equity_curve[-1]['equity']
        daily_return = (equity / prev_equity - 1) if prev_equity > 0 else 0.0
        
        # Update equity
        self.equity = equity
        
        # Record equity
        self.equity_curve.append({
            'date': date,
            'capital': self.capital,
            'equity': equity,
            'positions': len(self.positions),
            'returns': daily_return * 100  # Convert to percentage
        })
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.equity_curve:
            return {}
        
        try:
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('date', inplace=True)
            
            # Basic metrics
            initial_equity = self.config["initial_capital"]
            final_equity = self.equity
            total_return = (final_equity / initial_equity - 1) * 100
            
            # Calculate daily returns
            equity_df['daily_returns'] = equity_df['equity'].pct_change()
            
            # Calculate metrics
            trading_days = len(equity_df)
            trading_days_per_year = self.config["trading_days_per_year"]
            
            # Calculate annualized return
            annual_return = ((final_equity / initial_equity) ** (trading_days_per_year / trading_days) - 1) * 100
            
            # Calculate volatility (annualized)
            volatility = equity_df['daily_returns'].std() * np.sqrt(trading_days_per_year) * 100
            
            # Calculate drawdown
            equity_df['cumulative_max'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] / equity_df['cumulative_max'] - 1) * 100
            max_drawdown = equity_df['drawdown'].min()
            
            # Calculate Sharpe ratio
            risk_free_rate = self.config["risk_free_rate"]
            daily_risk_free = ((1 + risk_free_rate) ** (1 / trading_days_per_year) - 1)
            excess_returns = equity_df['daily_returns'] - daily_risk_free
            sharpe_ratio = (excess_returns.mean() / equity_df['daily_returns'].std()) * np.sqrt(trading_days_per_year)
            
            # Calculate Sortino ratio (only negative returns for denominator)
            negative_returns = equity_df['daily_returns'][equity_df['daily_returns'] < 0]
            sortino_ratio = 0.0
            if not negative_returns.empty:
                sortino_ratio = (equity_df['daily_returns'].mean() - daily_risk_free) / negative_returns.std() * np.sqrt(trading_days_per_year)
            
            # Calculate win rate
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                win_trades = trades_df[trades_df['action'] == 'sell']['profit_loss'] > 0
                win_rate = win_trades.sum() / win_trades.count() * 100 if win_trades.count() > 0 else 0.0
                
                # Calculate average profit/loss
                avg_profit = trades_df[trades_df['action'] == 'sell']['profit_loss'].mean()
                avg_profit_pct = trades_df[trades_df['action'] == 'sell']['profit_loss_pct'].mean()
                
                # Calculate average holding period
                avg_days_held = trades_df[trades_df['action'] == 'sell']['days_held'].mean()
            else:
                win_rate = 0.0
                avg_profit = 0.0
                avg_profit_pct = 0.0
                avg_days_held = 0.0
            
            # Collect metrics
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_profit_pct': avg_profit_pct,
                'avg_days_held': avg_days_held,
                'total_trades': len(self.trades) // 2 if self.trades else 0,  # Divide by 2 because each trade has buy and sell
                'trading_days': trading_days
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """
        Plot equity curve
        
        Args:
            save_path: Path to save the plot
        """
        try:
            if not self.equity_curve:
                logger.warning("No equity curve data to plot")
                return
            
            # Convert to DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('date', inplace=True)
            
            # Create figure
            plt.figure(figsize=(14, 8))
            
            # Plot equity curve
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(equity_df.index, equity_df['equity'], label='Equity')
            ax1.set_title('Equity Curve')
            ax1.set_ylabel('VND')
            ax1.grid(True)
            ax1.legend()
            
            # Plot drawdown
            equity_df['cumulative_max'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['equity'] / equity_df['cumulative_max'] - 1) * 100
            
            ax2 = plt.subplot(2, 1, 2, sharex=ax1)
            ax2.fill_between(equity_df.index, 0, equity_df['drawdown'], color='red', alpha=0.3)
            ax2.plot(equity_df.index, equity_df['drawdown'], color='red', label='Drawdown')
            ax2.set_title('Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.set_ylim(equity_df['drawdown'].min() * 1.1, 1)  # Extra 10% for better visualization
            ax2.grid(True)
            ax2.legend()
            
            # Add performance metrics text
            if self.performance_metrics:
                metrics_text = (
                    f"Total Return: {self.performance_metrics['total_return']:.2f}%\n"
                    f"Annual Return: {self.performance_metrics['annual_return']:.2f}%\n"
                    f"Volatility: {self.performance_metrics['volatility']:.2f}%\n"
                    f"Max Drawdown: {self.performance_metrics['max_drawdown']:.2f}%\n"
                    f"Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}\n"
                    f"Win Rate: {self.performance_metrics['win_rate']:.2f}%"
                )
                plt.figtext(0.01, 0.01, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Equity curve plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
    
    def plot_trade_analysis(self, save_path: Optional[str] = None) -> None:
        """
        Plot trade analysis
        
        Args:
            save_path: Path to save the plot
        """
        try:
            if not self.trades:
                logger.warning("No trade data to plot")
                return
            
            # Convert to DataFrame
            trades_df = pd.DataFrame(self.trades)
            
            # Filter sell trades
            sell_trades = trades_df[trades_df['action'] == 'sell'].copy()
            
            if sell_trades.empty:
                logger.warning("No completed trades to analyze")
                return
            
            # Create figure
            plt.figure(figsize=(14, 12))
            
            # 1. Plot profit/loss distribution
            ax1 = plt.subplot(2, 2, 1)
            sns.histplot(sell_trades['profit_loss_pct'], bins=20, kde=True, ax=ax1)
            ax1.set_title('Profit/Loss Distribution')
            ax1.set_xlabel('Profit/Loss (%)')
            ax1.set_ylabel('Count')
            ax1.axvline(0, color='r', linestyle='--')
            
            # 2. Plot profit/loss by symbol
            ax2 = plt.subplot(2, 2, 2)
            symbol_pnl = sell_trades.groupby('symbol')['profit_loss_pct'].mean().sort_values(ascending=False)
            symbol_pnl.plot(kind='bar', ax=ax2)
            ax2.set_title('Average Profit/Loss by Symbol')
            ax2.set_xlabel('Symbol')
            ax2.set_ylabel('Average Profit/Loss (%)')
            ax2.axhline(0, color='r', linestyle='--')
            
            # 3. Plot profit/loss vs holding period
            ax3 = plt.subplot(2, 2, 3)
            ax3.scatter(sell_trades['days_held'], sell_trades['profit_loss_pct'])
            ax3.set_title('Profit/Loss vs Holding Period')
            ax3.set_xlabel('Holding Period (days)')
            ax3.set_ylabel('Profit/Loss (%)')
            ax3.axhline(0, color='r', linestyle='--')
            ax3.grid(True)
            
            # 4. Plot cumulative profit/loss
            ax4 = plt.subplot(2, 2, 4)
            sell_trades.sort_values('date', inplace=True)
            sell_trades['cumulative_pnl'] = sell_trades['profit_loss'].cumsum()
            ax4.plot(sell_trades['date'], sell_trades['cumulative_pnl'])
            ax4.set_title('Cumulative Profit/Loss')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Cumulative Profit/Loss (VND)')
            ax4.grid(True)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Trade analysis plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting trade analysis: {str(e)}")
    
    def save_results(self, result_dir: str) -> bool:
        """
        Save backtest results to directory
        
        Args:
            result_dir: Directory to save results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(result_dir, exist_ok=True)
            
            # Save equity curve
            if self.equity_curve:
                equity_df = pd.DataFrame(self.equity_curve)
                equity_df.to_csv(os.path.join(result_dir, 'equity_curve.csv'), index=False)
            
            # Save trades
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                trades_df.to_csv(os.path.join(result_dir, 'trades.csv'), index=False)
            
            # Save signals
            if self.signals:
                signals_df = pd.DataFrame(self.signals)
                signals_df.to_csv(os.path.join(result_dir, 'signals.csv'), index=False)
            
            # Save performance metrics
            if self.performance_metrics:
                with open(os.path.join(result_dir, 'performance_metrics.json'), 'w') as f:
                    json.dump(self.performance_metrics, f, indent=4)
            
            # Save configuration
            with open(os.path.join(result_dir, 'config.json'), 'w') as f:
                json.dump(self.config, f, indent=4)
            
            # Save plots
            self.plot_equity_curve(os.path.join(result_dir, 'equity_curve.png'))
            self.plot_trade_analysis(os.path.join(result_dir, 'trade_analysis.png'))
            
            logger.info(f"Backtest results saved to {result_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving backtest results: {str(e)}")
            return False
    
    def generate_report(self, result_dir: str) -> str:
        """
        Generate HTML report
        
        Args:
            result_dir: Directory to save report
            
        Returns:
            Path to HTML report
        """
        try:
            import jinja2
            
            # Create report directory
            os.makedirs(result_dir, exist_ok=True)
            
            # Save results
            self.save_results(result_dir)
            
            # Create HTML report
            report_path = os.path.join(result_dir, 'report.html')
            
            # Define HTML template
            template_str = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Backtest Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1, h2 { color: #333; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
                    th { background-color: #f2f2f2; }
                    tr:hover { background-color: #f5f5f5; }
                    .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); grid-gap: 20px; margin-bottom: 20px; }
                    .metric-box { border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                    .metric-name { font-weight: bold; color: #666; }
                    .metric-value { font-size: 24px; color: #333; margin: 10px 0; }
                    .positive { color: green; }
                    .negative { color: red; }
                    .plot-container { margin: 20px 0; text-align: center; }
                    .plot-container img { max-width: 100%; height: auto; }
                </style>
            </head>
            <body>
                <h1>Backtest Report</h1>
                <p>Generated on: {{ generation_date }}</p>
                
                <h2>Performance Summary</h2>
                <div class="metrics-grid">
                    <div class="metric-box">
                        <div class="metric-name">Total Return</div>
                        <div class="metric-value {% if metrics.total_return > 0 %}positive{% else %}negative{% endif %}">
                            {{ '{:.2f}'.format(metrics.total_return) }}%
                        </div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-name">Annual Return</div>
                        <div class="metric-value {% if metrics.annual_return > 0 %}positive{% else %}negative{% endif %}">
                            {{ '{:.2f}'.format(metrics.annual_return) }}%
                        </div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-name">Sharpe Ratio</div>
                        <div class="metric-value {% if metrics.sharpe_ratio > 1 %}positive{% else %}negative{% endif %}">
                            {{ '{:.2f}'.format(metrics.sharpe_ratio) }}
                        </div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-name">Max Drawdown</div>
                        <div class="metric-value negative">
                            {{ '{:.2f}'.format(metrics.max_drawdown) }}%
                        </div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-name">Win Rate</div>
                        <div class="metric-value">
                            {{ '{:.2f}'.format(metrics.win_rate) }}%
                        </div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-name">Avg Profit/Loss</div>
                        <div class="metric-value {% if metrics.avg_profit_pct > 0 %}positive{% else %}negative{% endif %}">
                            {{ '{:.2f}'.format(metrics.avg_profit_pct) }}%
                        </div>
                    </div>
                </div>
                
                <h2>Equity Curve</h2>
                <div class="plot-container">
                    <img src="equity_curve.png" alt="Equity Curve">
                </div>
                
                <h2>Trade Analysis</h2>
                <div class="plot-container">
                    <img src="trade_analysis.png" alt="Trade Analysis">
                </div>
                
                <h2>Top Trades</h2>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Symbol</th>
                        <th>Action</th>
                        <th>Price</th>
                        <th>Quantity</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Days Held</th>
                    </tr>
                    {% for trade in top_trades %}
                    <tr>
                        <td>{{ trade.date.strftime('%Y-%m-%d') }}</td>
                        <td>{{ trade.symbol }}</td>
                        <td>{{ trade.action }}</td>
                        <td>{{ '{:,.0f}'.format(trade.price) }}</td>
                        <td>{{ '{:,}'.format(trade.quantity) }}</td>
                        <td class="{% if trade.profit_loss > 0 %}positive{% else %}negative{% endif %}">
                            {{ '{:,.0f}'.format(trade.profit_loss) }}
                        </td>
                        <td class="{% if trade.profit_loss_pct > 0 %}positive{% else %}negative{% endif %}">
                            {{ '{:.2f}'.format(trade.profit_loss_pct) }}%
                        </td>
                        <td>{{ '{:.1f}'.format(trade.days_held) }}</td>
                    </tr>
                    {% endfor %}
                </table>
                
                <h2>Configuration</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    {% for key, value in config.items() %}
                    <tr>
                        <td>{{ key }}</td>
                        <td>{{ value }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </body>
            </html>
            """
            
            # Prepare data for template
            if self.trades:
                trades_df = pd.DataFrame(self.trades)
                sell_trades = trades_df[trades_df['action'] == 'sell'].copy()
                
                # Get top 10 trades by profit/loss
                top_trades = sell_trades.sort_values('profit_loss', ascending=False).head(10)
            else:
                top_trades = pd.DataFrame()
            
            # Create template
            template = jinja2.Template(template_str)
            
            # Render template
            html = template.render(
                generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                metrics=self.performance_metrics,
                config=self.config,
                top_trades=top_trades.to_dict('records') if not top_trades.empty else []
            )
            
            # Save HTML report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            logger.info(f"HTML report generated at {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return ""
    
    def run_prophet_backtest(self, forecast_df, price_df, threshold=0.01):
        """
        Backtest đơn giản với output Prophet:
        - Nếu yhat > giá hiện tại * (1+threshold): tín hiệu mua
        - Nếu yhat < giá hiện tại * (1-threshold): tín hiệu bán
        - Ngược lại: giữ
        """
        results = []
        for i, row in forecast_df.iterrows():
            current_price = price_df.loc[row['ds'], 'close'] if row['ds'] in price_df.index else None
            if current_price is None:
                continue
            signal = 'hold'
            if row['yhat'] > current_price * (1 + threshold):
                signal = 'buy'
            elif row['yhat'] < current_price * (1 - threshold):
                signal = 'sell'
            results.append({
                'ds': row['ds'],
                'yhat': row['yhat'],
                'current_price': current_price,
                'signal': signal
            })
        return results 