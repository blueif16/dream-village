# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Interactive Memory Browser

Professional interactive browser for Memora memory stores and ChromaDB collections.
"""

import os
import sys
import json
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from .chroma_browser import ChromaBrowser, ChromaDocument
from ..utils.log import configure_logging

logger = logging.getLogger(__name__)


class InteractiveMemoryBrowser:
    """
    Interactive browser for Memora memory stores with professional CLI interface.
    
    Provides comprehensive exploration capabilities for ChromaDB-based memory storage
    with search, filtering, analysis, and export functionality.
    """
    
    def __init__(self, db_path: str, collection_name: str = None):
        """
        Initialize the interactive browser.
        
        Args:
            db_path: Path to ChromaDB database directory
            collection_name: Name of collection to browse (optional)
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize ChromaDB browser
        try:
            self.chroma_browser = ChromaBrowser(
                db_path=str(self.db_path),
                collection_name=collection_name
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB browser: {str(e)}")
            raise
            
        # Current state
        self.current_documents: List[ChromaDocument] = []
        self.current_filter: Optional[Dict[str, Any]] = None
        
    def run(self) -> None:
        """Run the interactive browser."""
        print("\n" + "=" * 70)
        print("🧠 MEMORA INTERACTIVE MEMORY BROWSER")
        print("=" * 70)
        print(f"📁 Database: {self.db_path}")
        print(f"📊 Collection: {self.chroma_browser.collection_name}")
        print("Type 'help' for available commands, 'quit' to exit")
        
        # Show initial stats
        self._show_quick_stats()
        
        while True:
            try:
                command = input("\n💭 Memory Browser > ").strip()
                
                if not command:
                    continue
                    
                # Parse command
                parts = command.split()
                cmd = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                if cmd in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                elif cmd == 'help':
                    self._show_help()
                elif cmd == 'stats':
                    self._show_detailed_stats()
                elif cmd == 'collections':
                    self._list_collections()
                elif cmd == 'switch':
                    self._switch_collection(args)
                elif cmd == 'list':
                    self._list_documents(args)
                elif cmd == 'search':
                    self._search_documents(args)
                elif cmd == 'filter':
                    self._filter_documents(args)
                elif cmd == 'clear':
                    self._clear_filter()
                elif cmd == 'show':
                    self._show_document_detail(args)
                elif cmd == 'export':
                    self._export_documents(args)
                elif cmd == 'count':
                    self._count_documents()
                elif cmd == 'metadata':
                    self._show_metadata_info(args)
                else:
                    print(f"❌ Unknown command: '{cmd}'. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                self.logger.error(f"Interactive browser error: {str(e)}")
                
    def _show_help(self) -> None:
        """Display help information."""
        print("\n📚 AVAILABLE COMMANDS:")
        print("-" * 50)
        print("🔍 Collection Management:")
        print("  collections           - List all collections")
        print("  switch <name>         - Switch to different collection")
        print("  stats                 - Show detailed collection statistics")
        print()
        print("📋 Document Browsing:")
        print("  list [limit]          - List documents (default: 10)")
        print("  count                 - Count total documents")
        print("  show <id|index>       - Show detailed document view")
        print()
        print("🔎 Search & Filter:")
        print("  search <query>        - Semantic search in documents")
        print("  filter <key>=<value>  - Filter by metadata")
        print("  clear                 - Clear current filters")
        print()
        print("📊 Analysis:")
        print("  metadata [key]        - Show metadata information")
        print("  stats                 - Detailed collection statistics")
        print()
        print("💾 Export:")
        print("  export <format> <path> - Export current documents")
        print("                          (formats: json, csv, txt)")
        print()
        print("ℹ️  System:")
        print("  help                  - Show this help message")
        print("  quit/exit/q           - Exit the browser")
        print()
        
        # Show current status
        filter_info = f" (filtered)" if self.current_filter else ""
        print(f"📊 Current Status: {len(self.current_documents)} documents loaded{filter_info}")
        
    def _show_quick_stats(self) -> None:
        """Show quick statistics."""
        try:
            stats = self.chroma_browser.get_collection_stats()
            print(f"📈 Quick Stats: {stats.total_documents:,} documents, {len(stats.metadata_keys)} metadata fields")
        except Exception as e:
            print(f"❌ Failed to load stats: {str(e)}")
            
    def _show_detailed_stats(self) -> None:
        """Show detailed collection statistics."""
        print("\n📊 COLLECTION STATISTICS")
        print("-" * 50)
        
        try:
            stats = self.chroma_browser.get_collection_stats()
            
            print(f"📋 Collection: {stats.collection_name}")
            print(f"📈 Total Documents: {stats.total_documents:,}")
            
            if stats.content_stats:
                print(f"\n📝 Content Statistics:")
                print(f"  Total Characters: {stats.content_stats.get('total_characters', 0):,}")
                print(f"  Average Length: {stats.content_stats.get('average_length', 0):.1f}")
                print(f"  Min Length: {stats.content_stats.get('min_length', 0)}")
                print(f"  Max Length: {stats.content_stats.get('max_length', 0)}")
                
            if stats.metadata_keys:
                print(f"\n🏷️  Metadata Fields ({len(stats.metadata_keys)}):")
                for key in sorted(stats.metadata_keys):
                    unique_count = stats.unique_metadata_values.get(key, 0)
                    print(f"  • {key}: {unique_count} unique values")
                    
        except Exception as e:
            print(f"❌ Failed to load detailed stats: {str(e)}")
            
    def _list_collections(self) -> None:
        """List all available collections."""
        print("\n📚 AVAILABLE COLLECTIONS")
        print("-" * 50)
        
        try:
            collections = self.chroma_browser.list_collections()
            
            if not collections:
                print("📭 No collections found")
                return
                
            current_name = self.chroma_browser.collection_name
            for i, collection in enumerate(collections, 1):
                marker = "👉" if collection == current_name else "  "
                print(f"{marker} {i}. {collection}")
                
        except Exception as e:
            print(f"❌ Failed to list collections: {str(e)}")
            
    def _switch_collection(self, args: List[str]) -> None:
        """Switch to a different collection."""
        if not args:
            print("❌ Usage: switch <collection_name>")
            return
            
        collection_name = args[0]
        
        try:
            if self.chroma_browser.switch_collection(collection_name):
                self.current_documents = []  # Clear cache
                self.current_filter = None
                print(f"✅ Switched to collection: {collection_name}")
                self._show_quick_stats()
            else:
                print(f"❌ Failed to switch to collection: {collection_name}")
                
        except Exception as e:
            print(f"❌ Switch failed: {str(e)}")
            
    def _list_documents(self, args: List[str]) -> None:
        """List documents in the collection."""
        limit = 10  # default
        
        if args:
            try:
                limit = int(args[0])
            except ValueError:
                print("❌ Invalid limit. Using default of 10.")
                
        print(f"\n📋 DOCUMENTS (showing up to {limit})")
        print("-" * 70)
        
        try:
            documents = self.chroma_browser.get_all_documents(limit=limit)
            
            if not documents:
                print("📭 No documents found")
                return
                
            for i, doc in enumerate(documents, 1):
                # Format metadata preview
                metadata_preview = ""
                if doc.metadata:
                    key_previews = []
                    for key, value in list(doc.metadata.items())[:2]:  # Show first 2 keys
                        key_previews.append(f"{key}={value}")
                    metadata_preview = " | " + ", ".join(key_previews)
                    if len(doc.metadata) > 2:
                        metadata_preview += "..."
                        
                content_preview = doc.content[:60] + "..." if len(doc.content) > 60 else doc.content
                
                print(f"{i:2d}. 🆔 {doc.id}")
                print(f"    📏 {len(doc.content)} chars{metadata_preview}")
                print(f"    💬 {content_preview}")
                print()
                
            self.current_documents = documents
            
        except Exception as e:
            print(f"❌ Failed to load documents: {str(e)}")
            
    def _search_documents(self, args: List[str]) -> None:
        """Search documents using semantic similarity."""
        if not args:
            print("❌ Usage: search <query>")
            return
            
        query = " ".join(args)
        print(f"\n🔍 SEARCH RESULTS for: '{query}'")
        print("-" * 70)
        
        try:
            documents = self.chroma_browser.search_documents(
                query=query,
                n_results=10
            )
            
            if not documents:
                print("📭 No matching documents found")
                return
                
            for i, doc in enumerate(documents, 1):
                # Show distance/similarity score
                score_info = f" (distance: {doc.distance:.3f})" if doc.distance is not None else ""
                
                # Find query terms in content for highlighting
                content = doc.content
                content_preview = content[:100] + "..." if len(content) > 100 else content
                
                print(f"{i:2d}. 🆔 {doc.id[:20]}...{score_info}")
                print(f"    📏 {len(doc.content)} chars")
                print(f"    💬 {content_preview}")
                print()
                
            self.current_documents = documents
            
        except Exception as e:
            print(f"❌ Search failed: {str(e)}")
            
    def _filter_documents(self, args: List[str]) -> None:
        """Filter documents by metadata."""
        if not args:
            print("❌ Usage: filter <key>=<value>")
            print("   Example: filter source_file=handbook.md")
            return
            
        filter_expr = " ".join(args)
        
        try:
            # Parse filter expression
            if "=" not in filter_expr:
                print("❌ Invalid filter format. Use: key=value")
                return
                
            key, value = filter_expr.split("=", 1)
            key = key.strip()
            value = value.strip()
            
            # Try to convert value to appropriate type
            if value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
                
            where_filter = {key: value}
            
            print(f"\n🔎 FILTERED RESULTS for: {key}={value}")
            print("-" * 70)
            
            documents = self.chroma_browser.filter_documents(
                where_filter=where_filter,
                limit=50
            )
            
            if not documents:
                print("📭 No documents match the filter")
                return
                
            for i, doc in enumerate(documents, 1):
                metadata_value = doc.metadata.get(key, "N/A")
                content_preview = doc.content[:60] + "..." if len(doc.content) > 60 else doc.content
                
                print(f"{i:2d}. 🆔 {doc.id}")
                print(f"    🏷️  {key}: {metadata_value}")
                print(f"    💬 {content_preview}")
                print()
                
            self.current_documents = documents
            self.current_filter = where_filter
            
        except Exception as e:
            print(f"❌ Filter failed: {str(e)}")
            
    def _clear_filter(self) -> None:
        """Clear current filter."""
        self.current_filter = None
        self.current_documents = []
        print("✅ Filter cleared")
        
    def _show_document_detail(self, args: List[str]) -> None:
        """Show detailed view of a specific document."""
        if not args:
            print("❌ Usage: show <document_id_or_index>")
            return
            
        identifier = args[0]
        document = None
        
        try:
            # Check if it's an index number
            if identifier.isdigit():
                index = int(identifier) - 1
                if 0 <= index < len(self.current_documents):
                    document = self.current_documents[index]
                else:
                    print(f"❌ Invalid index. Use 1-{len(self.current_documents)}")
                    return
            else:
                # Search by document ID
                all_docs = self.chroma_browser.get_all_documents()
                for doc in all_docs:
                    if doc.id == identifier or doc.id.startswith(identifier):
                        document = doc
                        break
                        
                if not document:
                    print(f"❌ Document not found: {identifier}")
                    return
            
            # Display detailed information
            print(f"\n🔍 DOCUMENT DETAILS")
            print("=" * 70)
            print(f"🆔 ID: {document.id}")
            print(f"📏 Content Length: {len(document.content)} characters")
            
            if document.distance is not None:
                print(f"📐 Distance: {document.distance:.6f}")
                
            if document.metadata:
                print(f"\n🏷️  METADATA:")
                print("-" * 40)
                for key, value in document.metadata.items():
                    print(f"  {key}: {value}")
                    
            print(f"\n💬 CONTENT:")
            print("-" * 70)
            print(f"[Index] {document.content}")
            print(f"[Value] {document.metadata.get('value', 'N/A')}")
            print("-" * 70)
            
        except Exception as e:
            print(f"❌ Failed to show document details: {str(e)}")
            
    def _export_documents(self, args: List[str]) -> None:
        """Export current documents."""
        if len(args) < 2:
            print("❌ Usage: export <format> <filepath>")
            print("   Formats: json, csv, txt")
            print("   Example: export json ./documents.json")
            return
            
        format_type, file_path = args[0], args[1]
        
        if format_type not in ['json', 'csv', 'txt']:
            print("❌ Supported formats: json, csv, txt")
            return
            
        if not self.current_documents:
            print("❌ No documents to export. Load some documents first.")
            return
            
        try:
            self.chroma_browser.export_documents(
                documents=self.current_documents,
                file_path=file_path,
                format=format_type
            )
            print(f"✅ Exported {len(self.current_documents)} documents to: {file_path}")
            
        except Exception as e:
            print(f"❌ Export failed: {str(e)}")
            
    def _count_documents(self) -> None:
        """Count total documents in collection."""
        try:
            stats = self.chroma_browser.get_collection_stats()
            print(f"📊 Total documents in collection: {stats.total_documents:,}")
        except Exception as e:
            print(f"❌ Failed to count documents: {str(e)}")
            
    def _show_metadata_info(self, args: List[str]) -> None:
        """Show metadata field information."""
        print("\n🏷️  METADATA INFORMATION")
        print("-" * 50)
        
        try:
            stats = self.chroma_browser.get_collection_stats()
            
            if not stats.metadata_keys:
                print("📭 No metadata fields found")
                return
                
            if args:
                # Show specific field
                field_name = args[0]
                if field_name in stats.unique_metadata_values:
                    unique_count = stats.unique_metadata_values[field_name]
                    print(f"📊 Field: {field_name}")
                    print(f"   Unique values: {unique_count}")
                    
                    # Get sample values
                    documents = self.chroma_browser.get_all_documents(limit=50)
                    values = set()
                    for doc in documents:
                        if field_name in doc.metadata:
                            values.add(str(doc.metadata[field_name]))
                            if len(values) >= 10:  # Show max 10 sample values
                                break
                                
                    if values:
                        print(f"   Sample values: {', '.join(list(values)[:10])}")
                else:
                    print(f"❌ Metadata field not found: {field_name}")
            else:
                # Show all fields
                print(f"📊 Total metadata fields: {len(stats.metadata_keys)}")
                print()
                for field in sorted(stats.metadata_keys):
                    unique_count = stats.unique_metadata_values.get(field, 0)
                    print(f"  • {field}: {unique_count} unique values")
                    
        except Exception as e:
            print(f"❌ Failed to load metadata info: {str(e)}")


def main():
    """Command-line interface for the interactive memory browser."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Memora Interactive Memory Browser - Explore ChromaDB collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Browse default collection in database
  python -m memora.browser ./memory_store
  
  # Browse specific collection
  python -m memora.browser ./memory_store --collection agent_memory
  
  # Browse with verbose logging
  python -m memora.browser ./memory_store --verbose
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
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
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
        # Initialize and run browser
        browser = InteractiveMemoryBrowser(
            db_path=args.db_path,
            collection_name=args.collection
        )
        browser.run()
        
    except Exception as e:
        print(f"❌ Failed to start memory browser: {str(e)}")
        logger.error(f"Memory browser startup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()