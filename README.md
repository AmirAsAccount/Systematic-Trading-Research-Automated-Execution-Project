# Systematic Trading Research & Automated Execution

A quantitative research and automated execution project exploring systematic trading strategies across **equities, centralized cryptocurrency markets, and decentralized exchanges**.

The project began as an open-ended attempt to determine whether market behavior could be converted into a systematic, automated trading process. Rather than committing to a single strategy, I developed and tested a range of hypotheses across different market structures, then progressively narrowed the research based on **statistical performance, transaction costs, execution constraints, and out-of-sample behavior**.

> **This repository documents both successful and failed research.** Strategies were not retained simply because they produced attractive backtests; many were rejected after accounting for execution, fees, latency, or out-of-sample performance.

---

## Research Process

The project follows an iterative research process:

```text
Hypothesis
    ↓
Data Collection
    ↓
Experimental Implementation
    ↓
Backtesting
    ↓
Transaction & Execution Analysis
    ↓
Out-of-Sample Evaluation
    ↓
Reject / Refine
    ↓
Architecture
    ↓
Automated Execution
```

The detailed development history is documented in [`process.md`](./process.md).

---

# Research Areas

## Equity Research

Research eventually concentrated on **low-float equities**, where limited available shares can amplify short-term demand and create substantial price movement.

Two primary approaches emerged:

### Catalyst Research

Investigated whether news events could be used to identify potentially large price movements in low-float stocks.

Research included:

* News/price correlation
* Catalyst classification
* Low-float screening
* Event-driven entry
* Post-catalyst behavior

This approach produced promising historical results but also exposed a major limitation: some of the largest historical moves were associated with information that could not realistically have been anticipated before the price movement.

### Momentum Research

This led to a second approach focused on the **price movement itself** rather than attempting to predict the catalyst.

The system evaluates stocks exhibiting significant gaps or momentum and attempts to identify continuation opportunities.

This became the basis for the current automated equity implementation.

---

# Cryptocurrency Research

Cryptocurrency provided a complementary research environment due to its:

* Continuous market hours
* High volatility
* Large number of tradable assets
* Rapid price dynamics
* Potential for high trade frequency

Research included:

* Order-flow dynamics
* Level 1 market data
* Hawkes processes
* Cawkes processes
* Point-process models
* Liquidity measurements
* Volatility behavior
* Dynamic signal construction
* Hyperparameter optimization

A number of approaches were rejected after accounting for transaction costs, execution latency, liquidity, or insufficient stability.

One later strategy produced an average gross backtest return of approximately **0.56% per trade**, with modeled post-fee performance around **0.30% per trade** over a large backtest.

This result is treated as a research finding rather than a claim of guaranteed profitability. Further testing proved the requirement in to determining how the edge behaves under realistic slippage, execution latency, position sizing, and live-market conditions.

---

# Decentralized Exchange Research

The earliest research focused on decentralized exchanges, particularly Solana-based markets and Jupiter.

Initial experiments investigated whether order-flow and liquidity information could provide short-term predictive signals.

The primary limitation was execution.

In highly volatile environments, the time required to generate and execute a signal could exceed the useful lifetime of the signal itself.

Additional constraints included:

* Network fees
* Platform fees
* Slippage
* Liquidity changes
* Execution latency
* Honeypots
* Rug-pull exposure

These results ultimately shifted the research toward centralized cryptocurrency markets and equities.

# Research Scope & Rejected Approaches

The research phase covered a much broader set of hypotheses than the current implementation. The goal was not to find a single indicator that worked historically, but to determine which forms of market information remained useful after costs, execution, timing, and regime differences were considered.

### Equity research

Research included:

* Low-float and small-cap screening
* News and catalyst/price correlation
* Event-driven entries
* Catalyst classification and catalyst strength
* Delayed post-catalyst reactions
* Gap-up and top-gainer behavior
* Momentum continuation
* Price levels and stop-level behavior
* Intraday seasonality and time-of-day effects
* RSI and OBV-style technical signals
* Volume and relative-volume relationships
* Amihud illiquidity and liquidity-adjusted signals
* Covariate-based models combining price, volume, liquidity, and event variables
* Predictive statistical models
* Bayesian probability estimates and conditional probability approaches
* Survival/time-to-event style modeling, including Cox-style prediction
* Signal combinations and feature-ranking approaches
* Historical backtesting of MASS-based signals and related composite signals

### Cryptocurrency research

Research included:

* Level 1 order-flow imbalance (OFI)
* Bid/ask pressure and volume imbalance
* Hawkes processes
* Cawkes processes
* General point-process models
* Event clustering and self-excitation
* Price/trade arrival dynamics
* Amihud-style liquidity measurements
* Volatility and realized-volatility features
* RSI, OBV, momentum, and volume-derived signals
* Dynamic and regime-dependent signals
* Covariate-based predictive models
* Bayesian probability and conditional-probability models
* Predictive/statistical models
* Optimization of signal parameters
* High-frequency entry/exit experiments
* Transaction-cost and execution-adjusted backtests

### Decentralized exchange research

DEX experiments investigated order flow, liquidity, transaction activity, execution timing, price impact, and short-horizon predictive signals in rapidly moving markets. These approaches were constrained heavily by fees, slippage, liquidity changes, network latency, and the useful lifetime of a signal.

### Why many academically motivated signals failed

A recurring result was that a signal described in the literature could be statistically meaningful in the market, dataset, horizon, and execution environment studied by the authors while being ineffective in this project.

That distinction became central to the research. Academic evidence establishes that a relationship can exist under particular conditions; it does not establish that the same relationship will survive a different asset universe, sampling frequency, market regime, trading horizon, liquidity profile, execution stack, or transaction-cost structure.

This was especially important for catalyst research. A catalyst or momentum driven event often represents **new information and therefore a potential regime change**, rather than an ordinary continuation of the historical process. A model trained on pre-event behavior can therefore be badly matched to the post-event state. The market may reprice immediately, liquidity may change, volatility may jump, and relationships estimated from ordinary periods may cease to describe the new regime.

The same issue appeared in microstructure research. OFI, point-process, liquidity, and volatility relationships can be real while still being economically unusable when the signal decays faster than the system can observe, decide, and execute.

### Major rejected or weakened approaches

The research ultimately rejected or substantially weakened:

* DEX strategies whose apparent edge disappeared after fees and slippage
* Level 1 OFI strategies whose predictive horizon was shorter than execution latency
* Hawkes/Cawkes and other point-process strategies without sufficiently stable out-of-sample performance
* Amihud illiquidity signals that did not translate cleanly across the project's assets and horizons
* Covariate and feature-combination models that failed to generalize
* Cox/time-to-event predictive approaches that did not provide a sufficiently robust trading edge
* Bayesian probability approaches whose estimated probabilities were sensitive to the underlying regime or assumptions
* RSI and OBV strategies that did not survive the project's filters and execution assumptions
* MASS/composite-signal backtests that did not provide sufficient evidence of durable edge
* Catalyst strategies whose largest winners depended on information that could not realistically have been known before the move
* Strategies whose performance was dominated by a small number of extreme winners
* Predictive models that performed well in historical samples but degraded under different regimes
* Approaches whose backtest profitability depended on data timing that could not be reproduced in live execution

These are not presented as proof that the underlying academic concepts are invalid. They are documented as **failed or insufficiently robust implementations under the conditions tested in this project**.

The important research conclusion was that profitability is conditional. A theoretically valid relationship can still fail as a trading strategy when the market, regime, horizon, liquidity, costs, or execution environment differs from the conditions under which the relationship was originally observed.

The objective therefore shifted toward identifying opportunities with a plausible asymmetric payoff and controlled downside, while treating low risk as a hypothesis to be demonstrated through position sizing, liquidity, execution, and risk controls rather than as an assumption.

---

# Current System

The current implementation combines market screening, real-time market data, news ingestion, signal generation, and automated execution.

### Data Sources

| Purpose                            | Technology            |
| ---------------------------------- | --------------------- |
| Stock screening / ticker discovery | FINVIZ API            |
| Market data                        | Schwab Developer API  |
| Automated execution                | Schwab Developer API  |
| Real-time news                     | Alpaca WebSocket      |
| Research / analysis                | Python, NumPy, Pandas |
| Optimization                       | Optuna                |

The architecture evolved in response to limitations discovered during live-system development.

---

## System Architecture

```text
                  ┌───────────────────┐
                  │  Market Screening │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │  Candidate Pool   │
                  └─────────┬─────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
      ┌───────────────┐           ┌───────────────┐
      │ Market Data   │           │   News Data   │
      │    Schwab     │           │     Alpaca    │
      └───────┬───────┘           └───────┬───────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
                  ┌───────────────────┐
                  │ Signal Evaluation │
                  └─────────┬─────────┘
                            ▼
                  ┌───────────────────┐
                  │ Risk / Execution  │
                  └─────────┬─────────┘
                            ▼
                  ┌───────────────────┐
                  │  Order Execution  │
                  └───────────────────┘
```

---

# Execution Engineering

Moving from backtesting to automated execution introduced problems that were not visible in the research environment.

The implementation therefore includes work around:

* Batch API requests
* Pagination
* API rate limits
* Connection management
* Session handling
* Threading and concurrency
* Request timeouts
* Quote retrieval
* Candidate scanning
* Data reuse
* Execution latency

One of the major architectural lessons was that **execution constraints can invalidate an otherwise promising strategy**.

Consequently, execution is treated as part of the trading system rather than as a separate implementation detail.

---

# Optimization & Validation

The project also includes systematic parameter optimization.

For example, the Ψ signal uses Optuna's **Tree-structured Parzen Estimator (TPE)** to explore signal parameters.

The optimization process uses temporal separation:

```text
60% ──────────────── 20% ──────────────── 20%
Train                  Validation              Test
   │                       │                    │
   │                       │                    │
Optimizer sees        Objective scored      Evaluated once
historical data       here                 after optimization
```

The validation period is used as the optimization objective rather than the final test period.

The final test set is reserved for evaluating whether the optimized strategy generalizes beyond the data used during development.

---

# Research Findings

The project has produced several broad findings.

### DEX markets

Potential signals were frequently overwhelmed by execution costs and latency.

### Centralized crypto

Higher-frequency research became more practical, although strategy stability and execution remain important constraints.

### Low-float equities

News catalysts and extreme price movements produced some of the strongest early backtest results.

### Catalyst strategies

The largest historical winners were not always predictable beforehand, leading to a shift toward directly observing momentum.

### Momentum

The resulting strategy path focuses on identifying abnormal price movement and attempting to capture continuation rather than predicting the fundamental catalyst responsible for the move.

---

# What Failed

Failed experiments are intentionally retained.

Examples include:

* DEX strategies that became unprofitable after fees
* Level 1 OFI approaches whose signal horizon was shorter than execution latency
* Point-process strategies without sufficiently stable performance
* Catalyst strategies dependent on unpredictable information
* Strategies whose performance was dominated by a small number of extreme winners
* Approaches that performed well in backtesting but encountered execution limitations

These experiments helped narrow the research space and influenced the architecture of the current system.

For the full development history, see [`process.md`](./process.md).

# Technology

**Languages & Libraries**

* Python
* NumPy
* Pandas
* Optuna
* CCXT
* Ollama

**Market Infrastructure**

* Schwab Developer API
* Alpaca WebSocket
* FINVIZ API
* Cryptocurrency exchange APIs

**Research Areas**

* Quantitative finance
* Time-series analysis
* Point processes
* Order-flow analysis
* Hyperparameter optimization
* Automated execution
* Market microstructure

---

# Project Status

**Completed**

This project progressed from exploratory quantitative research into a fully
implemented automated trading pipeline.

The research evaluated strategies across decentralized exchanges,
cryptocurrency markets, and equities, with particular attention to
transaction costs, execution latency, liquidity, and out-of-sample behavior.

The final implementation incorporated:

- Automated market screening
- Real-time market-data ingestion
- Real-time news ingestion
- Signal generation
- Risk controls
- Concurrent processing
- API session and connection management
- Automated order execution
- Backtesting and strategy evaluation

The project demonstrated that the viability of a trading strategy depends
not only on the underlying signal, but also on whether that signal can
survive transaction costs, latency, liquidity constraints, and real-world
execution.

The repository preserves the research process, including rejected strategies and the hypotheses that led to the final system. The experimental code itself is not treated as part of the production architecture.

> **Development Note:** This project was developed and tested primarily in a
> local environment throughout its development. The repository was published
> to GitHub after the project was completed, so the commit history does not
> represent the full development timeline. The repository is intended to
> preserve the completed implementation and research process rather than serve
> as a chronological development log.