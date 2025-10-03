# Changelog

All notable changes to the QQQ (Nasdaq-100) Change Alert System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-10-02

### Added
- **🏆 Top 10 Holdings Leaderboard** - Every email now includes a beautiful visual table of current top 10 stocks
  - Trophy emoji badges for podium positions (🥇🥈🥉)
  - Color-coded weight bars (green ≥5%, blue 2-5%, gray <2%)
  - Formatted market cap display ($XXX.XB or $X.XXT)
  - Always visible context, even when there are minimal changes
- New `format_top_holdings()` helper function for generating HTML leaderboard tables
- Comprehensive `.gitignore` file for Python projects
- `.gitkeep` files to preserve `data/cache/` and `data/raw/` directory structure
- Enhanced logging with emoji indicators (🚀 ✓ ⚠️ ❌) for better readability

### Changed
- Updated `send_email_alert()` to accept `current_data` parameter for leaderboard display
- Modified `check_for_changes()` to return both changes and current data
- Enhanced CSS styling in email templates for leaderboard table
- Improved email layout with leaderboard section positioned after stats grid

### Fixed
- **Pandas 2.0+ Compatibility** - Replaced deprecated `error_bad_lines=False` with `on_bad_lines='skip'`
- **Array Boolean Ambiguity** - Fixed "truth value of array" errors in weight processing
- Safe numeric conversions using `pd.to_numeric(errors='coerce')` throughout
- Comprehensive null/NaN handling in weight and market cap comparisons
- Enhanced symbol standardization with better error recovery

### Improved
- **Smart Content Detection** - Detects CSV vs Excel format before parsing attempts
- **Prioritized CSV Parsing** - QQQ data is actually CSV, so try CSV first
- **Reduced Redundant Parsing** - Early exit when CSV parsing succeeds
- **Better Error Messages** - More specific failure context and debugging info
- Enhanced validation with null safety in `validate_qqq_data()`
- Improved weight normalization with comprehensive error handling

## [2.0.0] - 2025-09-16

### Added
- QQQ-only focus with simplified codebase
- Enhanced data validation with detailed quality reports
- Comprehensive null safety throughout the pipeline
- Smart caching system (24-hour cache) for API responses
- Raw data preservation and analysis artifacts

### Changed
- **BREAKING**: Removed S&P 500 tracking functionality entirely
- Renamed GitHub Actions workflow to "Daily QQQ Change Check"
- Updated all documentation to reflect QQQ-only focus
- Simplified main script by removing dual-index complexity
- Enhanced logging with progress indicators and detailed validation

### Removed
- All S&P 500 related functions and data files
- `download_spy_holdings()` function
- `get_sp500_components()` and `get_sp500_components_hardcoded()` functions
- `data/sp500_current.json` file

### Fixed
- Improved Excel parsing with multiple fallback strategies
- Better column detection for varying ETF file formats
- Enhanced weight format detection and normalization
- More robust error handling with graceful degradation

## [1.5.0] - 2025-05-17

### Added
- Multi-strategy ETF file parsing (Excel, CSV, multiple encodings)
- ETF Excel Converter tool for local analysis
- ETF Holdings Analyzer for development and testing
- Raw file preservation for debugging

### Changed
- Enhanced weight normalization with intelligent format detection
- Improved symbol standardization and cleaning
- Better market cap integration with Alpha Vantage API

### Fixed
- Excel file format detection issues
- Weight percentage vs decimal confusion
- Column name variations in ETF provider files

## [1.0.0] - 2025-05-13

### Added
- Initial release with dual S&P 500 and QQQ tracking
- GitHub Actions workflow for daily automated checks
- HTML email alerts with change categorization
- Real-time market cap data integration (Alpha Vantage)
- Smart caching to reduce API usage
- Comprehensive change detection:
  - Index additions and removals
  - Rank changes
  - Weight changes (>0.1%)
  - Market cap changes (>5%)
- Top 20 position priority alerting
- Official ETF provider data integration:
  - State Street (SPY) for S&P 500
  - Invesco (QQQ) for Nasdaq-100

### Features
- Daily monitoring at 6:00 PM UTC (after US market close)
- Email notifications with detailed change breakdowns
- Data archiving for historical analysis
- Fallback mechanisms for data reliability
- Free to run using GitHub Actions

---

## Version History Summary

- **v2.1.0** (Current) - Top 10 Leaderboard + Enhanced Data Quality
- **v2.0.0** - QQQ-Only Focus with Major Simplification
- **v1.5.0** - Enhanced Parsing and Validation
- **v1.0.0** - Initial Dual-Index Tracking System

## Migration Notes

### Upgrading from v2.0.0 to v2.1.0
- No breaking changes
- New leaderboard feature automatically included in emails
- Update environment variables if needed (no new requirements)
- Existing data files remain compatible

### Upgrading from v1.x to v2.0.0
- **BREAKING**: S&P 500 tracking removed
- Remove `sp500_current.json` if it exists
- Update GitHub Actions workflow name references
- Review and update any custom scripts that referenced S&P 500 functions
- Update documentation and configuration files

## Support

For issues, questions, or contributions:
- GitHub Issues: [index-change-alerts/issues](https://github.com/YOUR_USERNAME/index-change-alerts/issues)
- Documentation: See README.md and CLAUDE.md
