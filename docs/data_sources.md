# Data Sources Configuration

## Deribit API

**Base URL**: https://www.deribit.com/api/v2

**Public Endpoints Used**:
- `/public/get_instruments` - List all BTC options
- `/public/ticker` - Get ticker with IV and Greeks
- `/public/get_order_book` - Order book snapshots
- `/public/get_index_price` - DVOL index

**Rate Limits**: 10 req/sec, burst 20

**Authentication**: Optional (public data), but API key enables higher limits

## Glassnode API

**Base URL**: https://api.glassnode.com/v1

**Metrics Used** (requires Top-Tier subscription):

### Exchange Flows
- `/metrics/transactions/transfers_volume_to_exchanges_sum`
- `/metrics/transactions/transfers_volume_from_exchanges_sum`  
- `/metrics/transactions/transfers_volume_exchanges_net`

### Supply Metrics
- `/metrics/distribution/balance_exchanges`
- `/metrics/supply/lth_sum` (Long-Term Holders)
- `/metrics/supply/sth_sum` (Short-Term Holders)
- `/metrics/supply/active_1y`
- `/metrics/supply/illiquid_sum`

### Dormancy & Age
- `/metrics/indicators/dormancy`
- `/metrics/indicators/cdd` (Coin Days Destroyed)
- `/metrics/indicators/soab` (Spent Output Age Bands)

### Stablecoins
- `/metrics/indicators/stablecoin_supply_ratio`
- `/metrics/transactions/transfers_volume_stablecoins_exchanges_net`

**Rate Limits**: 1 req/sec, burst 3

**Authentication**: API key required (set in GLASSNODE_API_KEY)

**Resolution**: Daily (24h) data by default

## Notes

- All endpoints configurable via `configs/data_sources.yaml`
- Implement robust retry logic with exponential backoff
- Cache responses (memory + file) to minimize API calls
- Store raw responses for audit trail
