# Monitoring System Integration

## Overview

The monitoring system tracks data quality and cache performance for the stock prediction platform in real-time. It provides insights into model training efficiency and data reliability.

## Components

### 1. Data Quality Monitor
Tracks data quality metrics for each ticker during model training:

- **Retention Rate**: Percentage of data retained after cleaning
- **Interpolation Detection**: Flags tickers with >40% interpolated data
- **Fallback Usage**: Tracks when extended lookback periods are needed

**Location**: `src/monitoring/data_quality_monitor.py`

### 2. Cache Monitor
Monitors model cache performance:

- **Cache Hit Rate**: Percentage of requests served from cache
- **Training Time**: Records time taken to train new models
- **Cache Efficiency**: Average training time and total cache statistics

**Location**: `src/monitoring/data_quality_monitor.py`

### 3. Performance Monitor
Logs predictions for future validation (placeholder):

- **Prediction Logging**: Records predictions with timestamps
- **Accuracy Tracking**: Framework for comparing predictions vs actuals

**Location**: `src/monitoring/data_quality_monitor.py`

## Integration Points

### webapp.py Integration

**Cache Monitoring** (Lines 739-740, 921-923):
```python
# Cache hit
cache_monitor = get_cache_monitor()
cache_monitor.record_cache_hit()

# Cache miss with timing
training_time = time.time() - training_start
cache_monitor.record_cache_miss(training_time)
```

**Data Quality Monitoring** (Lines 817-826):
```python
data_monitor = get_data_quality_monitor()
quality_metrics = {
    'original_rows': initial_len,
    'final_rows': len(data),
    'interpolation_percentage': (dropped_rows / initial_len * 100),
    'retention_rate': (len(data) / initial_len * 100),
    'used_fallback': False
}
data_monitor.log_data_quality(ticker, quality_metrics)
```

## API Endpoint

### GET `/api/monitoring/summary`

Returns comprehensive monitoring statistics.

**Response Example**:
```json
{
  "cache_performance": {
    "avg_training_time_seconds": 2.12,
    "cache_hits": 5,
    "cache_misses": 3,
    "hit_rate_percent": 62.5,
    "total_training_time_seconds": 6.36
  },
  "data_quality": {
    "average_retention": 85.3,
    "fallback_usage_count": 1,
    "high_interpolation_count": 2,
    "total_tickers_monitored": 8
  },
  "timestamp": "2025-11-19T19:20:16.893561"
}
```

**Usage**:
```bash
curl http://localhost:5000/api/monitoring/summary
```

## Metrics Explained

### Data Quality Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **average_retention** | Average % of data retained after cleaning | <50% (warning) |
| **high_interpolation_count** | Count of tickers with >40% interpolated data | >0 (review) |
| **fallback_usage_count** | Times extended lookback was needed | Monitor trend |
| **total_tickers_monitored** | Unique tickers tracked | N/A |

### Cache Performance Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **hit_rate_percent** | % of requests served from cache | >70% (optimal) |
| **avg_training_time_seconds** | Average time to train a model | <5s (good) |
| **cache_hits** | Total cache hits | Maximize |
| **cache_misses** | Total cache misses | Minimize |

## Alert Conditions

### High Interpolation Alert
- **Trigger**: Interpolation percentage >40%
- **Impact**: Predictions may be less reliable
- **Action**: Review data source quality for affected ticker

### Low Retention Alert
- **Trigger**: Retention rate <50%
- **Impact**: Insufficient training data
- **Action**: Increase lookback period or check data availability

### Slow Training Alert
- **Trigger**: Training time >10 seconds
- **Impact**: Poor user experience
- **Action**: Optimize feature engineering or model complexity

## Monitoring Best Practices

1. **Check Daily**: Review `/api/monitoring/summary` daily for trends
2. **Alert Thresholds**: Set up automated alerts for critical metrics
3. **Data Quality Review**: Investigate tickers with high interpolation
4. **Cache Optimization**: Monitor hit rate to validate caching strategy
5. **Performance Tracking**: Track training times to identify bottlenecks

## Future Enhancements

See `docs/IMPLEMENTATION_GUIDE.md` for advanced features:

- **Advanced Interpolation**: Cubic spline, ARIMA-based gap filling
- **Quality Scoring**: A-F grading system with confidence intervals
- **Prediction Validation**: Accuracy tracking over time
- **Export Reports**: Daily/weekly monitoring summaries

## Troubleshooting

### Monitoring Data Not Updating
- Verify server restart after integration
- Check logs for import errors
- Ensure monitoring imports at top of webapp.py

### High Interpolation Count
- Normal for commodities with sparse data (e.g., TIO=F)
- Review data source for affected tickers
- Consider extended lookback period

### Low Cache Hit Rate
- Expected during initial startup (cold cache)
- Improves as users request predictions
- Clear cache causes temporary drop

## Related Documentation

- `IMPLEMENTATION_GUIDE.md`: Full integration guide for future enhancements
- `IRON_ORE_FIX.md`: Technical details on data handling improvements
- API documentation at `/api/health` for server status
