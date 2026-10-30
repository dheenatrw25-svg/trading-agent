# Four-Hour BTC Advisory Playbook

This playbook guides a probabilistic four-hour directional forecast. It is not a
trading instruction or an automated trading system. Never describe a forecast as
an order to buy or sell.

## Data standards

- Use only completed four-hour BTC candles.
- Use the supplied OHLCV-based Volume Profile levels: `poc_price`,
  `value_area_high` (VAH), and `value_area_low` (VAL). This profile distributes
  each candle's volume to its typical price and is an approximation, not an
  exact exchange trade-level volume profile.
- `latest_completed_candle` contains the last complete candle's actual OHLCV.
- `recent_completed_candles` contains up to 20 timestamped completed four-hour
  candles for checking candle-close sequences such as sweeps and reclaims.
- Recent swing high/low context is the highest actual high and lowest actual low
  among the latest 20 completed four-hour candles. These are observed range
  extremes, not guaranteed liquidity.
- Volume comparison uses the latest completed candle against the median volume
  of up to 20 prior completed candles. Operational interpretation: expanding is
  at least 1.5x baseline; flat is 0.8x through 1.2x; declining is below 0.8x;
  other values are moderately above baseline. These are transparent heuristics,
  not validated or proven trading thresholds. If fewer than five prior candles
  exist or the baseline is invalid, mark the comparison unavailable.
- A single Binance open-interest snapshot cannot show a trend. Use only a
  validated change calculated from two timestamped historical observations.
  Changes above +1% are operationally rising, below -1% falling, and from -1%
  through +1% stagnant. These are heuristics, not proven thresholds. If
  historical observations are missing, stale, malformed, or unreliable, state
  that open-interest trend is unavailable and do not infer one.

## A. Volume Profile and market structure

- **Balanced market:** When price is inside the value area and volume is
  relatively flat, identify the market as balanced and favor a move toward POC.
  Above POC, the mean-reversion bias is DOWN; below POC, it is UP.
- **Bullish breakout:** Recognize an UP breakout only when a completed four-hour
  candle closes above VAH and volume expands. Use actual upper swing or liquidity
  levels as targets. Proposed invalidation is below POC.
- **VAL sweep and reclaim:** If a candle trades below VAL and subsequently
  closes back inside the value area, recognize a possible bullish reversal.
  Target POC first and VAH second. Invalidation is below the actual sweep low.
- **VAH rejection:** If price tests VAH and clearly rejects it, with derivatives
  evidence supporting exhausted bullish positioning, recognize a possible DOWN
  setup. Target POC first and VAL second. Invalidation is above the actual
  rejection high.
- A wick alone does not confirm a breakout, reclaim, or rejection. Require
  completed-candle close evidence and describe missing confirmation.

## B. Derivatives and positioning

- Use Binance Futures funding and open-interest data in the supplied context.
- If annualized funding exceeds 25%, do not chase an UP breakout at VAH. Reduce
  bullish confidence by at least two points or identify a plausible bearish
  mean-reversion scenario.
- Negative funding while price holds above POC or VAL supports an UP scenario
  only when price structure also confirms it.
- If price rises toward VAH while volume declines and validated open interest is
  stagnant, treat the movement as potentially weak rather than automatically
  bullish.
- Never label open interest rising, falling, or stagnant from its current
  snapshot alone.

## C. Macro filter

Use `dxy_daily_change_pct` as context, not as an automatic direction rule:

- At or above +0.40%: strong-dollar headwind. UP confidence must be no higher
  than 5/10.
- At or below -0.40%: supportive macro evidence for an UP scenario, provided
  price structure supports it.
- Between -0.15% and +0.15%: approximately neutral for this strategy.
- Other values are contextual and should not independently determine direction.

## D. Targets, invalidation, and risk/reward

- Ground every target, invalidation, and liquidity level in actual supplied
  market data: POC, VAH, VAL, completed four-hour candle highs/lows, or observed
  recent swing highs/lows. Never invent a level.
- Calculate reward/risk from current BTC price, target-range midpoint, and
  invalidation. For UP, risk is current price minus invalidation and reward is
  target midpoint minus current price. For DOWN, risk is invalidation minus
  current price and reward is current price minus target midpoint.
- An actionable setup requires positive risk and reward and reward/risk of at
  least 1.5:1.
- If levels are invalid, data is missing, or reward/risk is below 1.5:1, retain
  the best-supported UP or DOWN directional forecast, reduce confidence to 4/10
  or lower, and explicitly state that the forecast is not an actionable setup.
- When evidence is unclear or contradictory and no meaningful edge exists,
  choose the best-supported direction, normally use confidence 4/10 or lower,
  and explain the uncertainty and non-actionable status.
