# BTC-USD Autonomous Forecasting Playbook

## 1. Global Structure Rules (Applies to 2H & 4H)
- **RULE-G1 (POC Bias):** Favor `UP` only when `btc_price_at_prediction` is above `poc_price` or bouncing cleanly off `value_area_low`. Favor `DOWN` when rejected below `poc_price` or `value_area_high`.
- **RULE-G2 (Completed Candle Priority):** Use only completed candle closes. A wick alone does not confirm a breakout, reclaim, or rejection.
- **RULE-G3 (Volume Profile Context):** Treat `poc_price` as the strongest concentration zone, `value_area_high` as overhead resistance, and `value_area_low` as support within the active value area.
- **RULE-G4 (Directional Discipline):** When evidence is mixed, choose the best-supported direction only when a clear structure remains; otherwise abstain.

## 2. 2-Hour (2H) Horizon Rules
- **RULE-2H-01 (Momentum Confirmation):** Require immediate 15m candle body expansion in the forecast direction. Do not take a 2H trade if the last two 15m candles have long opposing wicks.
- **RULE-2H-02 (Trend Follow Through):** Only continue an `UP` or `DOWN` 2H thesis when the latest completed 15m candles continue the same directional impulse rather than stalling at the value-area edge.
- **RULE-2H-03 (Fast Reversal Filter):** If price compresses inside the value area with shrinking momentum and no clean directional close, reduce confidence and prefer `ABSTAIN`.
- **RULE-2H-04 (Target Validation):** A 2H target must be grounded in the active value area or recent directional candle structure. Never invent a target that is unsupported by the displayed chart.

## 3. 4-Hour (4H) Horizon Rules
- **RULE-4H-01 (Structural Clearance):** Do not forecast `UP` on a 4H horizon if `value_area_high` sits directly above `btc_price_at_prediction` without a confirmed candle close above it.
- **RULE-4H-02 (Macro Structure Constraint):** 4H trades require a stronger structural break or rejection than 2H trades; if the market is merely oscillating within the value area, avoid forcing a directional forecast.
- **RULE-4H-03 (Breakout Quality):** A 4H `UP` forecast is valid only if price closes above the active value-area ceiling with supportive volume and no immediate failure at the upper range.
- **RULE-4H-04 (Rejection Quality):** A 4H `DOWN` forecast is valid only when price rejects near value-area resistance and closes back within the value area or below `poc_price`.

## 4. Invalidated Setups (Abstain Filters)
- **RULE-ABS-01 (Flat Consolidation):** Output `ABSTAIN` if price is compressed tightly around `poc_price` and the latest candles show no decisive expansion or directional closure.
- **RULE-ABS-02 (Conflicting Momentum):** Output `ABSTAIN` if 15m structure, value-area placement, and directionality all conflict without one side clearly dominating.
- **RULE-ABS-03 (Weak Confirmation):** Output `ABSTAIN` if the trade idea depends on a single wick, a vague reclaim, or a level that has not been confirmed by a completed candle close.
- **RULE-ABS-04 (No Edge):** Output `ABSTAIN` when the forecast is directionally plausible but unsupported by a clear target, invalidation, or statistical edge relative to the chart structure.

## 5. Decision Schema
- `ABOVE_TARGET`: Directional call expecting price to move above the target within the active horizon.
- `BELOW_TARGET`: Directional call expecting price to fall below the target within the active horizon.
- `ABSTAIN`: No actionable edge is supported by completed candle structure, volume profile context, or clean confirmation.

## 6. Execution Principle
This playbook is a forecasting guide, not a command to buy or sell. The agent may issue a directional forecast only when the chart, value-area structure, and candle confirmation align with the rules above. Otherwise, it must abstain.
