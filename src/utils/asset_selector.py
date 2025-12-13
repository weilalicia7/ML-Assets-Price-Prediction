"""
Asset Selector - Interactive domain/sector selection for users.
Allows users to choose which domains (tech, oil, real estate, etc.) to analyze.
"""

import yaml
from typing import List, Dict, Set
import os


class AssetSelector:
    """
    Interactive asset selection by domain/sector.

    Allows users to choose from predefined sectors or create custom selections.
    """

    def __init__(self, config_path: str = 'config/assets.yaml'):
        """
        Initialize AssetSelector.

        Args:
            config_path: Path to assets configuration YAML file
        """
        self.config_path = config_path
        self.assets = self._load_assets()

    def _load_assets(self) -> Dict:
        """Load assets configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Assets config not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def list_categories(self) -> List[str]:
        """
        Get list of main categories (stocks, crypto, indices, etc.).

        Returns:
            List of category names
        """
        return list(self.assets.keys())

    def list_domains(self, category: str = 'stocks') -> List[str]:
        """
        Get list of domains/sectors within a category.

        Args:
            category: Category name (e.g., 'stocks', 'crypto')

        Returns:
            List of domain names
        """
        if category not in self.assets:
            raise ValueError(f"Unknown category: {category}")

        return list(self.assets[category].keys())

    def get_assets(self, category: str, domain: str) -> List[str]:
        """
        Get asset tickers for a specific domain.

        Args:
            category: Category name (e.g., 'stocks')
            domain: Domain name (e.g., 'technology', 'oil_energy')

        Returns:
            List of ticker symbols
        """
        if category not in self.assets:
            raise ValueError(f"Unknown category: {category}")

        if domain not in self.assets[category]:
            raise ValueError(f"Unknown domain '{domain}' in category '{category}'")

        return self.assets[category][domain]

    def get_preset(self, preset_name: str) -> List[str]:
        """
        Get predefined preset selection.

        Args:
            preset_name: Name of preset (e.g., 'tech_focus', 'diversified')

        Returns:
            List of ticker symbols
        """
        if 'presets' not in self.assets:
            raise ValueError("No presets defined in config")

        if preset_name not in self.assets['presets']:
            raise ValueError(f"Unknown preset: {preset_name}")

        return self.assets['presets'][preset_name]

    def list_presets(self) -> List[str]:
        """
        Get list of available presets.

        Returns:
            List of preset names
        """
        if 'presets' not in self.assets:
            return []
        return list(self.assets['presets'].keys())

    def select_interactive(self) -> List[str]:
        """
        Interactive selection of assets.

        Returns:
            List of selected ticker symbols
        """
        print("\n" + "="*60)
        print("ASSET SELECTOR")
        print("="*60)

        # Ask for selection mode
        print("\nSelect mode:")
        print("1. Use preset")
        print("2. Select by category/domain")
        print("3. Custom list")

        mode = input("\nEnter choice (1-3): ").strip()

        if mode == '1':
            return self._select_preset()
        elif mode == '2':
            return self._select_by_domain()
        elif mode == '3':
            return self._custom_selection()
        else:
            print("Invalid choice. Using default preset.")
            return self.get_preset('diversified')

    def _select_preset(self) -> List[str]:
        """Interactive preset selection."""
        presets = self.list_presets()

        print("\nAvailable presets:")
        for i, preset in enumerate(presets, 1):
            assets = self.get_preset(preset)
            print(f"{i}. {preset} ({len(assets)} assets): {', '.join(assets[:5])}{'...' if len(assets) > 5 else ''}")

        choice = input("\nEnter preset number: ").strip()

        try:
            idx = int(choice) - 1
            preset_name = presets[idx]
            assets = self.get_preset(preset_name)
            print(f"\n✓ Selected preset '{preset_name}' with {len(assets)} assets")
            return assets
        except (ValueError, IndexError):
            print("Invalid choice. Using 'diversified' preset.")
            return self.get_preset('diversified')

    def _select_by_domain(self) -> List[str]:
        """Interactive domain-based selection."""
        selected_assets = set()

        print("\nAvailable categories:")
        categories = [cat for cat in self.list_categories() if cat != 'presets']
        for i, cat in enumerate(categories, 1):
            print(f"{i}. {cat}")

        cat_choice = input("\nEnter category number: ").strip()

        try:
            category = categories[int(cat_choice) - 1]
        except (ValueError, IndexError):
            print("Invalid choice. Using 'stocks'.")
            category = 'stocks'

        print(f"\nAvailable domains in '{category}':")
        domains = self.list_domains(category)
        for i, domain in enumerate(domains, 1):
            assets = self.get_assets(category, domain)
            print(f"{i}. {domain} ({len(assets)} assets)")

        domain_choices = input("\nEnter domain numbers (comma-separated, e.g., '1,3,5'): ").strip()

        for choice in domain_choices.split(','):
            try:
                idx = int(choice.strip()) - 1
                domain = domains[idx]
                assets = self.get_assets(category, domain)
                selected_assets.update(assets)
                print(f"✓ Added {len(assets)} assets from '{domain}'")
            except (ValueError, IndexError):
                print(f"✗ Invalid choice: {choice}")

        result = list(selected_assets)
        print(f"\n✓ Total selected: {len(result)} unique assets")
        return result

    def _custom_selection(self) -> List[str]:
        """Custom ticker input."""
        print("\nEnter ticker symbols separated by commas:")
        print("Example: AAPL, MSFT, BTC-USD, XOM")

        tickers = input("\nTickers: ").strip()
        assets = [t.strip().upper() for t in tickers.split(',')]

        print(f"\n✓ Selected {len(assets)} assets: {', '.join(assets)}")
        return assets

    def select_from_config(self, selection_config: Dict) -> List[str]:
        """
        Select assets based on configuration dictionary.

        Args:
            selection_config: Dict with selection criteria
                Examples:
                - {'preset': 'tech_focus'}
                - {'category': 'stocks', 'domains': ['technology', 'semiconductors']}
                - {'tickers': ['AAPL', 'BTC-USD']}

        Returns:
            List of ticker symbols
        """
        selected = set()

        # Preset selection
        if 'preset' in selection_config:
            preset = selection_config['preset']
            selected.update(self.get_preset(preset))

        # Category/domain selection
        if 'category' in selection_config and 'domains' in selection_config:
            category = selection_config['category']
            domains = selection_config['domains']

            for domain in domains:
                assets = self.get_assets(category, domain)
                selected.update(assets)

        # Direct ticker selection
        if 'tickers' in selection_config:
            selected.update(selection_config['tickers'])

        return list(selected)

    def get_domain_info(self, category: str, domain: str) -> Dict:
        """
        Get detailed information about a domain.

        Args:
            category: Category name
            domain: Domain name

        Returns:
            Dictionary with domain information
        """
        assets = self.get_assets(category, domain)

        return {
            'category': category,
            'domain': domain,
            'asset_count': len(assets),
            'assets': assets
        }

    def print_summary(self, assets: List[str]) -> None:
        """
        Print summary of selected assets.

        Args:
            assets: List of ticker symbols
        """
        print("\n" + "="*60)
        print("ASSET SELECTION SUMMARY")
        print("="*60)

        # Categorize assets
        stocks = [a for a in assets if '-USD' not in a and not a.startswith('^') and '=F' not in a]
        crypto = [a for a in assets if '-USD' in a]
        indices = [a for a in assets if a.startswith('^')]
        commodities = [a for a in assets if '=F' in a]

        print(f"\nTotal assets: {len(assets)}")
        if stocks:
            print(f"  Stocks: {len(stocks)} - {', '.join(stocks[:5])}{'...' if len(stocks) > 5 else ''}")
        if crypto:
            print(f"  Crypto: {len(crypto)} - {', '.join(crypto)}")
        if indices:
            print(f"  Indices: {len(indices)} - {', '.join(indices)}")
        if commodities:
            print(f"  Commodities: {len(commodities)} - {', '.join(commodities)}")

        print("="*60)


def main():
    """
    Example usage of AssetSelector.
    """
    selector = AssetSelector()

    # Example 1: Interactive selection
    # assets = selector.select_interactive()

    # Example 2: Preset selection
    print("Example 1: Using preset")
    assets = selector.get_preset('tech_focus')
    selector.print_summary(assets)

    # Example 3: Domain selection
    print("\n\nExample 2: Selecting specific domains")
    tech_assets = selector.get_assets('stocks', 'technology')
    crypto_assets = selector.get_assets('crypto', 'major_coins')
    combined = list(set(tech_assets + crypto_assets))
    selector.print_summary(combined)

    # Example 4: Config-based selection
    print("\n\nExample 3: Config-based selection")
    config = {
        'category': 'stocks',
        'domains': ['semiconductors', 'oil_energy']
    }
    assets = selector.select_from_config(config)
    selector.print_summary(assets)

    # Example 5: Multiple categories
    print("\n\nExample 4: Multiple domains")
    selection = selector.select_from_config({
        'category': 'stocks',
        'domains': ['technology', 'semiconductors', 'real_estate']
    })
    selector.print_summary(selection)


if __name__ == "__main__":
    main()
