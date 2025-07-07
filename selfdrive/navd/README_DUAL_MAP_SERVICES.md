# Dual Map Services Configuration

This implementation uses two separate map services for different purposes:

## MapTiler (Map Display)
- **Purpose**: Tile-based map rendering and visualization
- **Usage**: Used by `map_renderer.cc` for displaying the base map
- **Configuration**: 
  - Environment variable: `MAPTILER_TOKEN`
  - API endpoint: `https://api.maptiler.com`
  - Provider: QMapLibre with MapTilerProvider template

## Mapbox (Navigation Routing)
- **Purpose**: Route calculation and navigation instructions
- **Usage**: Used by `navd.py` for route planning and directions
- **Configuration**:
  - Environment variable: `MAPBOX_TOKEN`
  - API endpoint: `https://api.mapbox.com`
  - Used for Directions API v5

## Setup Instructions

### Required Environment Variables
```bash
export MAPTILER_TOKEN="your_maptiler_token_here"
export MAPBOX_TOKEN="your_mapbox_token_here"
```

### Getting API Keys
1. **MapTiler Token**: 
   - Sign up at https://www.maptiler.com/
   - Get your API key from the dashboard
   
2. **Mapbox Token**:
   - Sign up at https://www.mapbox.com/
   - Get your access token from the dashboard

### Why Both Services?
- **MapTiler**: Provides excellent tile-based map rendering with good performance for display
- **Mapbox**: Offers superior routing algorithms and traffic-aware navigation
- **Separation of Concerns**: Map display and route calculation are handled independently
- **Reliability**: If one service fails, the other can continue to function

## Files Modified
- `main.cc`: Updated to clarify service separation
- `map_renderer.cc`: Dedicated to MapTiler for map display
- `navd.py`: Dedicated to Mapbox for navigation routing
- `map_helpers.h/cc`: Updated constants and settings for MapTiler
