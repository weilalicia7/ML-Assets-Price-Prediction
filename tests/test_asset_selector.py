"""
Unit tests for asset selector module.
Tests domain-based asset selection functionality.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.asset_selector import AssetSelector


class TestAssetSelector:
    """Test cases for AssetSelector class."""

    def test_load_config(self):
        """Test loading asset configuration."""
        print("\n" + "="*60)
        print("TEST 1: Loading asset configuration")
        print("="*60)

        selector = AssetSelector(config_path='config/assets.yaml')

        categories = selector.list_categories()
        print(f"[OK] Loaded {len(categories)} categories: {categories}")

        assert len(categories) > 0, "Should have at least one category"
        assert 'stocks' in categories, "Should have stocks category"
        assert 'crypto' in categories, "Should have crypto category"

    def test_list_stock_domains(self):
        """Test listing stock domains."""
        print("\n" + "="*60)
        print("TEST 2: Listing stock domains")
        print("="*60)

        selector = AssetSelector(config_path='config/assets.yaml')
        domains = selector.list_domains('stocks')

        print(f"✓ Found {len(domains)} stock domains:")
        for domain in domains:
            print(f"  - {domain}")

        assert len(domains) > 0, "Should have stock domains"
        assert 'technology' in domains, "Should have technology domain"
        assert 'oil_energy' in domains, "Should have oil_energy domain"
        assert 'real_estate' in domains, "Should have real_estate domain"
        assert 'semiconductors' in domains, "Should have semiconductors domain"

    def test_get_technology_stocks(self):
        """Test getting technology sector stocks."""
        print("\n" + "="*60)
        print("TEST 3: Getting technology stocks")
        print("="*60)

        selector = AssetSelector(config_path='config/assets.yaml')
        tech_stocks = selector.get_assets('stocks', 'technology')

        print(f"✓ Found {len(tech_stocks)} tech stocks:")
        print(f"  {', '.join(tech_stocks)}")

        assert len(tech_stocks) > 0, "Should have tech stocks"
        assert 'AAPL' in tech_stocks, "Should include AAPL"
        assert 'MSFT' in tech_stocks, "Should include MSFT"
        assert 'GOOGL' in tech_stocks, "Should include GOOGL"

    def test_get_oil_energy_stocks(self):
        """Test getting oil/energy sector stocks."""
        print("\n" + "="*60)
        print("TEST 4: Getting oil/energy stocks")
        print("="*60)

        selector = AssetSelector(config_path='config/assets.yaml')
        oil_stocks = selector.get_assets('stocks', 'oil_energy')

        print(f"✓ Found {len(oil_stocks)} oil/energy stocks:")
        print(f"  {', '.join(oil_stocks)}")

        assert len(oil_stocks) > 0, "Should have oil stocks"
        assert 'XOM' in oil_stocks, "Should include XOM"
        assert 'CVX' in oil_stocks, "Should include CVX"

    def test_get_real_estate_stocks(self):
        """Test getting real estate sector stocks."""
        print("\n" + "="*60)
        print("TEST 5: Getting real estate stocks")
        print("="*60)

        selector = AssetSelector(config_path='config/assets.yaml')
        real_estate = selector.get_assets('stocks', 'real_estate')

        print(f"✓ Found {len(real_estate)} real estate stocks:")
        print(f"  {', '.join(real_estate)}")

        assert len(real_estate) > 0, "Should have real estate stocks"
        assert 'AMT' in real_estate or 'PLD' in real_estate, "Should include major REITs"

    def test_get_semiconductor_stocks(self):
        """Test getting semiconductor sector stocks."""
        print("\n" + "="*60)
        print("TEST 6: Getting semiconductor stocks")
        print("="*60)

        selector = AssetSelector(config_path='config/assets.yaml')
        chips = selector.get_assets('stocks', 'semiconductors')

        print(f"✓ Found {len(chips)} semiconductor stocks:")
        print(f"  {', '.join(chips)}")

        assert len(chips) > 0, "Should have chip stocks"
        assert 'NVDA' in chips, "Should include NVDA"
        assert 'AMD' in chips, "Should include AMD"
        assert 'INTC' in chips, "Should include INTC"

    def test_get_crypto_assets(self):
        """Test getting crypto assets."""
        print("\n" + "="*60)
        print("TEST 7: Getting cryptocurrency assets")
        print("="*60)

        selector = AssetSelector(config_path='config/assets.yaml')
        crypto = selector.get_assets('crypto', 'major_coins')

        print(f"✓ Found {len(crypto)} major cryptocurrencies:")
        print(f"  {', '.join(crypto)}")

        assert len(crypto) > 0, "Should have crypto assets"
        assert 'BTC-USD' in crypto, "Should include Bitcoin"
        assert 'ETH-USD' in crypto, "Should include Ethereum"

    def test_get_preset(self):
        """Test getting preset selections."""
        print("\n" + "="*60)
        print("TEST 8: Getting preset selections")
        print("="*60)

        selector = AssetSelector(config_path='config/assets.yaml')

        # Test tech focus preset
        tech_focus = selector.get_preset('tech_focus')
        print(f"✓ Tech focus preset: {', '.join(tech_focus)}")
        assert len(tech_focus) > 0, "Tech focus should have assets"

        # Test energy focus preset
        energy_focus = selector.get_preset('energy_focus')
        print(f"✓ Energy focus preset: {', '.join(energy_focus)}")
        assert len(energy_focus) > 0, "Energy focus should have assets"

        # Test diversified preset
        diversified = selector.get_preset('diversified')
        print(f"✓ Diversified preset: {', '.join(diversified)}")
        assert len(diversified) > 0, "Diversified should have assets"

    def test_config_based_selection(self):
        """Test configuration-based selection."""
        print("\n" + "="*60)
        print("TEST 9: Config-based multi-domain selection")
        print("="*60)

        selector = AssetSelector(config_path='config/assets.yaml')

        # Select technology and oil sectors
        config = {
            'category': 'stocks',
            'domains': ['technology', 'oil_energy', 'semiconductors']
        }

        assets = selector.select_from_config(config)
        selector.print_summary(assets)

        assert len(assets) > 0, "Should have selected assets"
        # Should have unique assets (no duplicates)
        assert len(assets) == len(set(assets)), "Should not have duplicates"

    def test_mixed_selection(self):
        """Test selecting from multiple categories."""
        print("\n" + "="*60)
        print("TEST 10: Mixed stocks and crypto selection")
        print("="*60)

        selector = AssetSelector(config_path='config/assets.yaml')

        tech = selector.get_assets('stocks', 'technology')
        crypto = selector.get_assets('crypto', 'major_coins')
        combined = list(set(tech + crypto))

        selector.print_summary(combined)

        assert len(combined) > 0, "Should have assets"
        # Check we have both types
        has_stocks = any('-USD' not in a for a in combined)
        has_crypto = any('-USD' in a for a in combined)
        assert has_stocks, "Should have stocks"
        assert has_crypto, "Should have crypto"

    def test_list_all_presets(self):
        """Test listing all available presets."""
        print("\n" + "="*60)
        print("TEST 11: Listing all presets")
        print("="*60)

        selector = AssetSelector(config_path='config/assets.yaml')
        presets = selector.list_presets()

        print(f"✓ Found {len(presets)} presets:")
        for preset in presets:
            assets = selector.get_preset(preset)
            print(f"  - {preset}: {len(assets)} assets")

        assert len(presets) > 0, "Should have presets"

    def test_domain_info(self):
        """Test getting domain information."""
        print("\n" + "="*60)
        print("TEST 12: Getting domain information")
        print("="*60)

        selector = AssetSelector(config_path='config/assets.yaml')
        info = selector.get_domain_info('stocks', 'technology')

        print(f"✓ Domain info:")
        print(f"  Category: {info['category']}")
        print(f"  Domain: {info['domain']}")
        print(f"  Asset count: {info['asset_count']}")
        print(f"  Assets: {', '.join(info['assets'][:5])}...")

        assert info['category'] == 'stocks'
        assert info['domain'] == 'technology'
        assert info['asset_count'] > 0


def run_all_tests():
    """Run all tests manually."""
    print("\n" + "="*80)
    print("RUNNING ASSET SELECTOR TESTS")
    print("="*80)

    tester = TestAssetSelector()

    tests = [
        ('Load config', tester.test_load_config),
        ('List stock domains', tester.test_list_stock_domains),
        ('Get technology stocks', tester.test_get_technology_stocks),
        ('Get oil/energy stocks', tester.test_get_oil_energy_stocks),
        ('Get real estate stocks', tester.test_get_real_estate_stocks),
        ('Get semiconductor stocks', tester.test_get_semiconductor_stocks),
        ('Get crypto assets', tester.test_get_crypto_assets),
        ('Get presets', tester.test_get_preset),
        ('Config-based selection', tester.test_config_based_selection),
        ('Mixed selection', tester.test_mixed_selection),
        ('List all presets', tester.test_list_all_presets),
        ('Domain info', tester.test_domain_info),
    ]

    passed = 0
    failed = 0

    for i, (name, test_func) in enumerate(tests, 1):
        try:
            test_func()
            print(f"\n✓ TEST {i} PASSED: {name}")
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST {i} FAILED: {name}")
            print(f"   Error: {str(e)}")
            failed += 1

    print("\n" + "="*80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed out of {len(tests)} total")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
