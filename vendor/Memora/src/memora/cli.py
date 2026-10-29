# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Memora CLI

Main command-line interface for the Memora memory framework.
"""

import sys
import argparse
import logging
from pathlib import Path

from .utils.log import configure_logging
from .browser import InteractiveMemoryBrowser, MemoryViewer

logger = logging.getLogger(__name__)


def browser_command(args):
    """Handle browser subcommand."""
    from .browser.__main__ import main as browser_main
    
    # Prepare args for browser main
    browser_args = [args.db_path]
    
    if args.collection:
        browser_args.extend(['-c', args.collection])
    if args.stats:
        browser_args.append('--stats')
    if args.analyze:
        browser_args.append('--analyze')
    if args.search:
        browser_args.extend(['--search', args.search])
    if args.output:
        browser_args.extend(['--output', args.output])
    if args.limit:
        browser_args.extend(['--limit', str(args.limit)])
    if args.verbose:
        browser_args.append('--verbose')
    if args.no_interactive:
        browser_args.append('--no-interactive')
    
    # Replace sys.argv and call browser main
    original_argv = sys.argv
    try:
        sys.argv = ['memora-browser'] + browser_args
        browser_main()
    finally:
        sys.argv = original_argv


def main():
    """Main entry point for Memora CLI."""
    parser = argparse.ArgumentParser(
        description="Memora - Advanced Memory Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive memory browser
  memora browser /path/to/memory_store
  
  # Browse specific collection
  memora browser /path/to/memory_store -c agent_memory
  
  # Quick statistics
  memora browser /path/to/memory_store --stats
  
  # Search memories
  memora browser /path/to/memory_store --search "incident management"
  
  # Generate analysis report
  memora browser /path/to/memory_store --analyze --output report.json
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Memora 0.1.0'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Browser subcommand
    browser_parser = subparsers.add_parser(
        'browser',
        help='Interactive memory browser',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive browser
  memora browser /path/to/memory_store
  
  # Browse specific collection
  memora browser /path/to/memory_store -c agent_memory
  
  # Quick statistics
  memora browser /path/to/memory_store --stats
  
  # Search and analyze
  memora browser /path/to/memory_store --search "procedures"
        """
    )
    
    browser_parser.add_argument(
        'db_path',
        help='Path to ChromaDB database directory'
    )
    
    browser_parser.add_argument(
        '--collection', '-c',
        help='Name of collection to browse'
    )
    
    browser_parser.add_argument(
        '--stats',
        action='store_true',
        help='Show quick statistics and exit'
    )
    
    browser_parser.add_argument(
        '--analyze',
        action='store_true',
        help='Generate comprehensive analysis report'
    )
    
    browser_parser.add_argument(
        '--search', '-s',
        help='Search query to analyze'
    )
    
    browser_parser.add_argument(
        '--output', '-o',
        help='Output file path for results'
    )
    
    browser_parser.add_argument(
        '--limit', '-l',
        type=int,
        default=10,
        help='Limit number of results (default: 10)'
    )
    
    browser_parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Disable interactive mode'
    )
    
    browser_parser.set_defaults(func=browser_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    configure_logging()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute command
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Command failed: {str(e)}")
            print(f"❌ Error: {str(e)}")
            sys.exit(1)
    else:
        # No subcommand provided
        parser.print_help()
        print("\n💡 Try: memora browser /path/to/memory_store")
        sys.exit(1)


if __name__ == "__main__":
    main()