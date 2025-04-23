# Stock Analysis Bot

A comprehensive stock analysis Telegram bot for the Vietnamese market, providing technical and fundamental analysis using AI-powered insights.

## Features

- **Technical Analysis**: Calculate indicators and detect patterns across multiple timeframes
- **Fundamental Analysis**: Analyze company financials and key metrics
- **AI-Powered Reports**: Generate comprehensive analysis reports using Claude and GPT models
- **Market News**: Fetch and filter latest market news
- **Model Training**: Automatically train Prophet and XGBoost models for price prediction
- **User Authentication**: Restrict access to approved users only
- **Caching System**: Redis caching for improved performance

## Project Structure

```
stock_bot/
├── app/
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── base_ai_analyzer.py     # Abstract base class for AI services
│   │   └── openrouter_analyzer.py  # OpenRouter implementation
│   ├── database/
│   │   ├── __init__.py
│   │   ├── db_manager.py           # Database management for user data and reports
│   │   ├── model_db_manager.py     # Database management for trained models
│   │   └── redis_manager.py        # Redis cache management
│   ├── models/
│   │   ├── __init__.py
│   │   └── database_models.py      # SQLAlchemy models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Load stock data from various sources
│   │   ├── data_pipeline.py        # Coordinate data processing workflow
│   │   ├── fundamental_analysis.py # Fundamental analysis calculations
│   │   ├── model_trainer.py        # Train and evaluate Prophet and XGBoost models
│   │   ├── news_service.py         # News fetching and processing
│   │   └── technical_analysis.py   # Technical indicators and pattern detection
│   ├── telegram/
│   │   ├── __init__.py
│   │   └── handlers.py             # Telegram command handlers
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py               # Configuration constants
│   │   ├── data_normalizer.py      # Data normalization and cleaning
│   │   └── helpers.py              # Utility functions
│   └── __init__.py
├── main.py                         # Application entry point
└── requirements.txt                # Python dependencies
```

## Setup and Configuration

### Requirements

- Python 3.8+
- PostgreSQL database
- Redis server
- Telegram Bot token

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Database Configuration
DATABASE_URL=postgresql+asyncpg://username:password@localhost/stockbot

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_CACHE_EXPIRY=86400

# Telegram Configuration
TELEGRAM_TOKEN=your_telegram_bot_token
ADMIN_USER_IDS=123456789,987654321

# AI API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key

# Webhook Configuration (for production)
WEBHOOK_URL=https://your-domain.com/webhook
WEBHOOK_PORT=8443
WEBHOOK_LISTEN=0.0.0.0
```

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables
4. Run the bot:
   ```
   python main.py
   ```

### Running Tests

```
python main.py test
```

## Usage

Start a chat with your bot on Telegram and use the following commands:

- `/start` - Get started with the bot
- `/analyze [symbol]` - Analyze a stock symbol (e.g., `/analyze FPT`)
- `/id` - Get your Telegram user ID
- `/approve [user_id]` - (Admin only) Approve a user by ID

## Credits

This bot uses the following libraries and services:
- Python-Telegram-Bot
- SQLAlchemy
- Redis
- Prophet
- XGBoost
- TA-Lib
- OpenRouter AI API
- pandas, numpy, and other data analysis tools

## License

This project is licensed under the MIT License - see the LICENSE file for details. 