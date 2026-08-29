# Research & Development Process

## 1. Initial Objective

This project began with a deliberately broad objective:

> Develop an automated trading pipeline capable of identifying and executing profitable opportunities across different markets and strategies.

I quickly recognized that this was too broad to approach as a single implementation problem. Rather than committing to one strategy prematurely, I treated the project as a research process: identify market structures, develop hypotheses, build experimental tests, evaluate the results, and eliminate approaches that failed under realistic constraints.

Because the project began with relatively small amounts of capital, I initially focused on highly volatile markets where small positions could potentially generate meaningful returns.

The initial research therefore centered heavily around cryptocurrency and decentralized exchanges.



# 2. Decentralized Exchange Research

## 2.1 Initial DEX Research

My first major research direction involved decentralized exchanges, particularly the Solana ecosystem and Jupiter.

The initial hypothesis was that decentralized markets might provide opportunities that were difficult to exploit in traditional markets because of:

* High volatility
* Fragmented liquidity
* Rapid price movements
* On-chain transaction data
* Order-flow information
* Less efficient market structure

I began researching the mechanics of decentralized execution and built a series of experimental scripts to test different approaches.

The exploratory implementations were optimized for **research velocity rather than software architecture**. At this stage, the primary question was whether the underlying market behavior existed at all.



## 2.2 Order-Flow Imbalance

One of the first approaches investigated was a proxy for Order Flow Imbalance (OFI) using Level 1 market data.

The hypothesis was that short-term changes in buying and selling pressure could provide predictive information about near-term price movement.

The results were not sufficiently consistent.

The primary problem was not necessarily that order-flow information contained no information. Instead, the execution environment created several additional problems.

The time between signal generation and execution was often long relative to the speed of the underlying price movement. Cryptocurrency volatility could therefore exceed the time horizon over which the signal remained useful.

This produced significant exposure to:

* Slippage
* Execution latency
* Liquidity changes
* Adverse price movement between signal and execution
* Transaction costs

DEX-specific risks also introduced additional concerns, including malicious tokens, honeypots, and rug-pull exposure.

The resulting conclusion was that a theoretically useful signal could still be economically useless if the market moved faster than the system could execute.



## 2.3 Transaction Costs

Transaction costs became one of the earliest major constraints.

With small capital, fixed transaction costs represented a substantial percentage of potential returns.

A strategy could appear profitable before fees while becoming unprofitable after accounting for:

* Network fees
* Platform fees
* Swap costs
* Slippage
* Execution losses

This led to an important change in the research methodology:

> A strategy was not considered promising simply because its theoretical signal generated positive returns. Its expected return had to be evaluated against the actual costs and constraints of executing the strategy.

This eventually became one of the primary filters used throughout the project.



# 3. Centralized Cryptocurrency Research

After the limitations of DEX execution became apparent, I moved toward centralized cryptocurrency markets.

The objective was to retain some of the volatility of cryptocurrency while reducing the extreme execution constraints encountered on decentralized exchanges.

The hypothesis was that smaller cryptocurrency assets could provide enough volatility to generate short-term opportunities while maintaining sufficient liquidity for more practical execution.

I began testing a variety of quantitative approaches.



## 3.1 Point Processes

A significant portion of the research focused on point-process models.

I investigated:

* Hawkes processes
* Cawkes processes
* Other point-process formulations
* Combinations of point-process variables with market dynamics
* Order-flow-derived features

The goal was to determine whether clustered market events could provide predictive information about subsequent price behavior.

These experiments produced interesting statistical behavior but did not consistently produce a sufficiently stable trading edge after accounting for execution requirements.

The experiments were nevertheless useful because they demonstrated that a mathematically interesting relationship does not necessarily translate into a profitable trading strategy.



## 3.2 Broader Quantitative Research

The exploratory phase covered a large range of quantitative approaches across cryptocurrency, equities, and decentralized exchanges. The implementations were deliberately experimental: the objective was to test whether a measurable relationship existed before investing in a production architecture around it.

The research included:

* Order Flow Imbalance (OFI) and Level 1 order-flow proxies
* Hawkes and Cawkes processes
* General point-process and event-clustering models
* Price/trade arrival dynamics
* Amihud illiquidity and liquidity-adjusted signals
* Covariates combining price, volume, liquidity, volatility, and event information
* Predictive statistical models
* Bayesian probability and conditional-probability approaches
* Cox-style time-to-event / hazard prediction
* RSI and OBV technical signals
* Volume and relative-volume signals
* MASS/composite signal backtesting
* Price-level and stop-level models
* Momentum and gap behavior
* Intraday/time-of-day effects
* News and catalyst correlation
* Catalyst classification and event-driven models
* Dynamic and regime-dependent signals
* Hyperparameter optimization and machine-learning-assisted signal selection

The important result was not that every model failed. Rather, the research repeatedly showed that **statistical significance, predictive accuracy, and trading profitability are different questions**.

A relationship can be real while its economic value is too small, too short-lived, too unstable, or too expensive to trade.

### Why literature-backed signals did not always transfer

Several experiments were motivated by published research. Those papers were valuable because they provided evidence that particular relationships can exist under defined conditions. They were not treated as guarantees that the same relationship would work in this project.

The conditions often differed materially:

* Different asset classes and exchanges
* Different liquidity and market-cap distributions
* Different sampling frequencies
* Different holding periods
* Different transaction costs
* Different execution latency
* Different market microstructure
* Different volatility regimes
* Different participant composition
* Different data quality and availability
* Different signal definitions and preprocessing

This became especially important for **new-information events**. A catalyst can create a new market regime rather than simply continue the historical regime. The arrival of an unexpected FDA decision, acquisition announcement, financing event, or other major catalyst can simultaneously change price, volatility, liquidity, order flow, and participant behavior.

Consequently, a model estimated from the pre-event distribution can become poorly specified immediately after the event. This helped explain why some catalyst-related historical relationships looked extremely strong while being difficult to exploit prospectively.

The same principle appeared in high-frequency research. Published order-flow or point-process relationships may operate at a particular time scale, while the project's end-to-end signal-to-order latency can be longer than the useful predictive horizon. In that situation, the relationship can be statistically real and still produce no practical edge.

### Rejected or insufficiently robust approaches

The following approaches were rejected, weakened, or failed to provide enough evidence for inclusion in the final strategy:

| Approach | Market(s) | Main issue encountered |
|  |  |  |
| DEX order-flow / liquidity strategies | Crypto / DEX | Fees, slippage, liquidity changes, execution speed |
| Level 1 OFI | Crypto | Signal decay relative to execution latency |
| Hawkes processes | Crypto / high-frequency markets | Interesting event clustering without sufficiently stable tradable performance |
| Cawkes / related point processes | Crypto | Statistical structure did not consistently survive trading constraints |
| General point-process models | Crypto / equities | Predictive behavior did not reliably become an executable edge |
| Amihud illiquidity | Crypto / equities | Relationship was highly dependent on asset, horizon, and liquidity regime |
| Covariate models | Crypto / equities | Feature relationships did not generalize reliably |
| Bayesian probability models | Crypto / equities | Estimates were sensitive to assumptions and regime changes |
| Predictive models | Crypto / equities | Historical predictive power did not consistently translate to live-executable returns |
| Cox / hazard-style prediction | Crypto / equities | Time-to-event prediction did not produce sufficiently robust trade selection |
| RSI | Crypto / equities | Conventional signal behavior was not sufficiently strong after project-specific filters and costs |
| OBV | Crypto / equities | Volume-price relationship did not provide a durable standalone edge |
| MASS/composite signals | Crypto / equities | Backtest performance was not sufficiently robust for production use |
| Price/stop-level models | Crypto / equities | Relationships varied materially across regimes |
| News/catalyst prediction | Equities | Largest moves often depended on information unavailable before the move |
| Catalyst continuation models | Equities | Post-event regime changes made historical relationships difficult to transfer |
| Gap/top-gainer models | Equities | Required additional filters to distinguish continuation from exhaustion |

These failures are important because they narrowed the research space. They also prevented the final system from becoming a collection of indicators selected solely because they looked good in historical data.



# 4. Discovery of Low-Float Equity Opportunities

The most stable results during the early research phase came from equities rather than cryptocurrency.

I became particularly interested in low-float stocks.

The hypothesis was based on the effect of limited available shares on short-term price movement.

When significant buying pressure enters a stock with a small float, relatively small changes in demand can produce disproportionately large price movements.

This created an attractive asymmetry for short-term strategies:

* Failed catalysts could result in limited or moderate downside.
* Successful catalysts could produce extremely large percentage moves.
* Low float could amplify demand-driven price movement.
* News events provided identifiable temporal anchors for research.

Backtesting around news events produced significantly more promising results than many of the earlier cryptocurrency experiments.

At this point, the project shifted from broad market exploration toward building an actual automated equity trading system.



# 5. Catalyst Strategy

The first major equity strategy focused on identifying low-float stocks experiencing significant news catalysts.

The basic research pipeline became:

```text
Market Screening
       ↓
Identify Low-Float Candidates
       ↓
Identify News Catalyst
       ↓
Evaluate Catalyst
       ↓
Estimate Expected Price Response
       ↓
Generate Entry
       ↓
Manage Position
       ↓
Exit
```

The research initially produced very promising backtest results.

However, deeper analysis revealed an important problem.

The largest returns were frequently associated with catalysts that could not realistically have been anticipated before the price movement.

Examples included events such as unexpected FDA decisions or other information that was difficult or impossible to acquire and act upon before the market reacted.

This created a form of survivorship in the apparent strategy performance.

A backtest could identify a huge move following a catalyst, but that did not necessarily mean an automated system could have purchased the stock before the move.



# 6. Catalyst Research Limitations

Further investigation revealed several problems with directly targeting catalysts.

### Unpredictable catalysts

Some of the largest historical moves resulted from information that could not reasonably be predicted beforehand.

### Delayed reactions

Some stocks experienced significant price increases well after the original catalyst.

This made it difficult to determine whether the catalyst itself was responsible for the trade opportunity.

### M&A ambiguity

Mergers and acquisitions introduced additional complications because the timing and market interpretation of the information could vary significantly.

### Top-gainer bias

When analyzing the largest historical winners, I found that some of the biggest moves did not fit the original catalyst hypothesis cleanly.

Some stocks had:

* No obvious catalyst
* A catalyst substantially earlier than the major price movement
* A delayed reaction
* Momentum that appeared to become self-reinforcing

This suggested that attempting to predict the fundamental reason for the move might be less useful than identifying the price behavior itself.

This observation led to the development of a second strategy path.



# 7. Path B — Momentum Research

Instead of attempting to predict which news events would produce large moves, Path B focused on identifying stocks that had **already demonstrated significant upward momentum**.

The underlying question changed from:

> "Which catalyst will cause a stock to move?"

to:

> "Which stocks are already moving significantly, and can the continuation of that movement be traded?"

The general research process became:

```text
Market Scan
      ↓
Identify Large Gap / Momentum
      ↓
Filter Candidates
      ↓
Measure Momentum Characteristics
      ↓
Evaluate Entry Conditions
      ↓
Execute
      ↓
Monitor Momentum
      ↓
Exit
```

This approach effectively changed the system from a **predictive catalyst strategy** into a **reactive momentum strategy**.

Rather than requiring the system to correctly interpret an unknown future catalyst, the system could observe the market response directly.

This reduced dependence on predicting news outcomes.



# 8. Data Architecture

Once the research produced strategies worth implementing, the project transitioned from research toward architecture.

The primary question became:

> What data does the system require, where does that data come from, and how can it be retrieved quickly enough for the strategy?

The system required several independent data sources.

### Screening and Tickers

I used an unofficial FINVIZ API implementation for screening and ticker discovery.

Because the screening process required large numbers of symbols, requests were organized around pagination and batch retrieval rather than repeatedly querying individual symbols.

### Market Data and Execution

Schwab's developer APIs were used for live market data and automated execution.

This introduced practical constraints around:

* Authentication
* API rate limits
* Quote retrieval
* Connection stability
* Request latency
* Session state

### News

Alpaca's WebSocket API was used to receive news information in real time.

This allowed news ingestion to operate independently from the primary market-data pipeline.



# 9. Initial Architecture Problems

The first implementations exposed several problems that were not apparent during strategy research.

A backtest assumes that data is available when needed.

A live system has to actually retrieve it.

This created bottlenecks involving:

* API latency
* Rate limits
* Concurrent requests
* Session management
* Threading
* Quote retrieval
* Authentication
* Candidate scanning

The system therefore required architectural changes independent of the trading strategy itself.



# 10. Batch Retrieval

One of the major architectural changes was replacing unnecessary repeated API calls with batched data retrieval.

Instead of repeatedly requesting information for individual symbols, the system began grouping requests where possible.

For screening, pagination was used to efficiently process larger symbol sets.

For market data, state was reused when information was already available rather than requesting the same information repeatedly.

This reduced unnecessary API traffic and improved the responsiveness of the scanner.



# 11. Session and Threading Improvements

As the system moved closer to live execution, session management became a significant issue.

Earlier versions could become blocked or delayed when individual API requests failed or timed out.

The architecture was modified to better separate:

* Market scanning
* Candidate evaluation
* Market-data retrieval
* News processing
* Execution

Threading was adjusted so that one slow or failed request would not unnecessarily block the rest of the trading pipeline.

Connection failures were also treated as bounded failures rather than reasons to indefinitely block the strategy.

This was important because in a live trading environment, **a delayed signal can be equivalent to a missed signal**.



# 12. Execution vs. Backtesting

Another major lesson from the project was the difference between a profitable backtest and an executable strategy.

The backtesting environment allowed me to evaluate the theoretical strategy.

The live implementation introduced additional variables:

* Network latency
* API latency
* Quote freshness
* Order execution
* Slippage
* Liquidity
* Position availability
* Rate limits

Therefore, the project increasingly treated execution as part of the strategy rather than as an independent implementation detail.

A strategy with a theoretical edge that disappears after realistic execution costs is not considered successful.


# 14. Optimization

As the research progressed, I began using systematic optimization rather than manually selecting parameters.

One example is the Ψ signal optimization framework.

The optimizer uses Optuna's Tree-structured Parzen Estimator (TPE) sampler to search across signal parameters.

The search includes parameters such as:

* Differencing windows
* Amihud liquidity windows
* Normalization windows
* Entry thresholds
* Exit thresholds
* Maximum holding periods

The evaluation uses a temporal split:

```text
First 60%
    ↓
Training / Optimization

Next 20%
    ↓
Validation / Objective

Final 20%
    ↓
Held-Out Test
```

The optimizer evaluates the validation period rather than optimizing directly on the final test set.

The test set is reserved for final evaluation.

This structure was introduced specifically to reduce the risk of optimizing directly against historical noise.



# 15. AI-Assisted Research

A significant portion of the experimental stage involved AI-assisted development.

During exploratory research, AI tools were used to accelerate the creation of testing scripts, strategy variations, debugging iterations, and experimental implementations.

This was particularly useful when the objective was to rapidly test a large number of hypotheses.

The experimental scripts therefore vary significantly in architecture, abstraction, and code quality.

The important distinction is that the AI-generated or AI-assisted scripts were used primarily as **research tools**.

The research questions, hypotheses, evaluation criteria, interpretation of results, and eventual architecture decisions remained part of the development process.

As the project transitioned from research into implementation, greater emphasis was placed on code organization, reliability, execution constraints, and maintainability.



# 16. Current Architecture

The current implementation represents the transition from exploratory research into an automated trading system.

The system broadly consists of:

```text
             ┌─────────────────┐
             │ Market Screening│
             └────────┬────────┘
                      ↓
             ┌─────────────────┐
             │ Candidate Pool  │
             └────────┬────────┘
                      ↓
        ┌─────────────┴─────────────┐
        ↓                           ↓
┌─────────────────┐        ┌─────────────────┐
│ Market Data     │        │ News Data       │
│ Schwab API      │        │ Alpaca WebSocket│
└────────┬────────┘        └────────┬────────┘
         └─────────────┬─────────────┘
                       ↓
              ┌─────────────────┐
              │ Signal Evaluation│
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Risk / Execution│
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Order Execution │
              └─────────────────┘
```

The architecture has been designed around the constraints discovered during research rather than around the assumptions of the initial strategy.



# 17. Current Research Direction

The project is currently divided into several research directions.

### Equity

The primary equity research focuses on:

* Low-float stocks
* Gap-up behavior
* Momentum continuation
* Catalyst-driven movement
* Execution latency
* Short-term price dynamics

### Cryptocurrency

Crypto research focuses on:

* High-frequency opportunities
* Point-process dynamics
* Order-flow relationships
* Liquidity
* Transaction costs
* Strategy frequency
* Execution scalability

### Optimization

Optimization research focuses on determining whether parameters can be selected systematically without producing excessive historical overfitting.



# 18. What Failed

Failure is intentionally retained as part of this project.

Several approaches that appeared mathematically or intuitively promising did not survive testing.

Examples include:

* DEX strategies whose returns disappeared after fees
* Level 1 OFI approaches whose predictive horizon was shorter than execution latency
* Point-process strategies without stable out-of-sample performance
* Catalyst strategies whose largest historical winners depended on information that could not realistically have been anticipated
* Strategies whose apparent profitability depended heavily on a small number of extreme winners
* Approaches that worked in backtesting but encountered execution limitations in live environments

These failures changed the direction of the project.

The purpose of retaining them is not to claim that every experiment succeeded.

It is to document how the research process progressively eliminated weak hypotheses.



# 19. Lessons Learned

Several general lessons have emerged from the project.

### 1. A profitable signal is not necessarily a profitable strategy.

Transaction costs, latency, liquidity, and slippage can completely eliminate a theoretical edge.

### 2. Backtesting is only the beginning.

A strategy must eventually survive increasingly realistic assumptions.

### 3. Market structure matters.

A strategy that works in equities may not work in cryptocurrency, and a strategy that works on a centralized exchange may not work on a DEX.

### 4. The largest historical returns can be misleading.

Extreme winners often require careful investigation to determine whether they could actually have been identified before the move.

### 5. Execution is part of the strategy.

Latency and data availability are not merely engineering problems. They directly affect whether the trading hypothesis is viable.

### 6. Failed experiments are useful.

A rejected strategy narrows the research space and provides information about what market behavior is unlikely to be exploitable.

### 7. Research and production code have different requirements.

During exploration, rapid experimentation is more important than abstraction.

Once a strategy survives research, architecture, reliability, testing, and maintainability become increasingly important.

