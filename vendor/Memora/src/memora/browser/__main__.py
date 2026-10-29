# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Memora Browser CLI

Command-line interface for Memora memory browser tools.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from .interactive_browser import InteractiveMemoryBrowser
from .memory_viewer import MemoryViewer
from ..utils.log import configure_logging

logger = logging.getLogger(__name__)


def main():
    """Main entry point for Memora browser CLI."""
    parser = argparse.ArgumentParser(
        description="Memora Memory Browser - Interactive exploration of ChromaDB memory stores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive browser
  python -m memora.browser /path/to/memory_store
  
  # Browse specific collection
  python -m memora.browser /path/to/memory_store -c agent_memory
  
  # Generate analysis report
  python -m memora.browser /path/to/memory_store --analyze --output report.json
  
  # Quick statistics
  python -m memora.browser /path/to/memory_store --stats
  
  # Search and analyze
  python -m memora.browser /path/to/memory_store --search "incident management"
        """
    )
    
    parser.add_argument(
        'db_path',
        help='Path to ChromaDB database directory'
    )
    
    parser.add_argument(
        '--collection', '-c',
        help='Name of collection to browse (if not specified, uses first available)'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        default=True,
        help='Launch interactive browser (default)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show quick statistics and exit'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Generate comprehensive analysis report'
    )
    
    parser.add_argument(
        '--search', '-s',
        help='Search query to analyze'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file path for analysis/search results'
    )
    
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=10,
        help='Limit number of results (default: 10)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive mode (for scripting)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate database path
    if not os.path.exists(args.db_path):
        print(f"❌ Database path does not exist: {args.db_path}")
        sys.exit(1)
        
    try:
        # Initialize viewer
        viewer = MemoryViewer(
            db_path=args.db_path,
            collection_name=args.collection
        )
        
        # Handle different modes
        if args.stats:
            # Quick statistics mode
            print(viewer.quick_stats())
            
        elif args.search:
            # Search mode
            print(f"🔍 Searching for: '{args.search}'")
            results = viewer.search_and_analyze(
                query=args.search,
                n_results=args.limit
            )
            
            if args.output:
                # Save to file
                import json
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"✅ Results saved to: {args.output}")
            else:
                # Print to console
                print(f"\n📊 Search Results:")
                print(f"Query: {results['query']}")
                print(f"Total Results: {results['total_results']}")
                
                if 'distance_stats' in results:
                    stats = results['distance_stats']
                    print(f"Distance Range: {stats['min']:.3f} - {stats['max']:.3f} (avg: {stats['average']:.3f})")
                
                print("\n📋 Top Results:")
                for i, result in enumerate(results.get('results', [])[:5], 1):
                    distance_info = f" (distance: {result['distance']:.3f})" if result['distance'] else ""
                    print(f"{i}. ID: {result['id']}{distance_info}")
                    print(f"   Content: {result['content_preview'][:100]}...")
                    print()
                    
        elif args.analyze:
            # Analysis mode
            print("📊 Generating comprehensive analysis...")
            
            output_path = args.output or f"memory_analysis_{viewer.chroma_browser.collection_name}.json"
            viewer.export_analysis_report(
                output_path=output_path,
                include_search_examples=True
            )
            print(f"✅ Analysis report saved to: {output_path}")
            
        elif not args.no_interactive:
            # Interactive mode (default)
            browser = InteractiveMemoryBrowser(
                db_path=args.db_path,
                collection_name=args.collection
            )
            browser.run()
        else:
            # Non-interactive mode - show summary
            summary = viewer.get_memory_summary(include_samples=True, sample_count=3)
            
            if args.output:
                import json
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f"✅ Summary saved to: {args.output}")
            else:
                print("📊 Memory Summary:")
                print(f"Collection: {summary['collection_name']}")
                stats = summary['statistics']
                print(f"Documents: {stats['total_documents']:,}")
                print(f"Metadata Fields: {len(stats['metadata_keys'])}")
                if stats['content_stats']:
                    print(f"Total Characters: {stats['content_stats']['total_characters']:,}")
                    print(f"Average Length: {stats['content_stats']['average_length']:.1f}")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        logger.error(f"CLI error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()