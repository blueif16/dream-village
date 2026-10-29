# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Memory Viewer

Non-interactive memory viewer for analysis and reporting.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from .chroma_browser import ChromaBrowser, ChromaDocument, ChromaStats

logger = logging.getLogger(__name__)


class MemoryViewer:
    """
    Non-interactive memory viewer for analysis and reporting.
    
    Provides programmatic access to memory analysis capabilities
    without interactive interface.
    """
    
    def __init__(self, db_path: str, collection_name: str = None):
        """
        Initialize the memory viewer.
        
        Args:
            db_path: Path to ChromaDB database directory
            collection_name: Name of collection to analyze (optional)
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
            
    def get_memory_summary(
        self, 
        include_samples: bool = True,
        sample_count: int = 5
    ) -> Dict[str, Any]:
        """
        Get comprehensive memory summary.
        
        Args:
            include_samples: Whether to include sample documents
            sample_count: Number of sample documents to include
            
        Returns:
            Dict[str, Any]: Memory summary
        """
        try:
            # Get collection statistics
            stats = self.chroma_browser.get_collection_stats()
            
            summary = {
                'database_path': str(self.db_path),
                'collection_name': stats.collection_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'statistics': asdict(stats)
            }
            
            # Add sample documents if requested
            if include_samples and stats.total_documents > 0:
                documents = self.chroma_browser.get_all_documents(limit=sample_count)
                summary['sample_documents'] = [
                    {
                        'id': doc.id,
                        'content_length': len(doc.content),
                        'content_preview': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        'metadata': doc.metadata
                    }
                    for doc in documents
                ]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate memory summary: {str(e)}")
            return {
                'database_path': str(self.db_path),
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
    def search_and_analyze(
        self, 
        query: str,
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search memories and provide analysis.
        
        Args:
            query: Search query
            n_results: Number of results to analyze
            
        Returns:
            Dict[str, Any]: Search results and analysis
        """
        try:
            # Perform search
            documents = self.chroma_browser.search_documents(
                query=query,
                n_results=n_results
            )
            
            # Analyze results
            analysis = {
                'query': query,
                'total_results': len(documents),
                'analysis_timestamp': datetime.now().isoformat(),
                'results': []
            }
            
            if documents:
                # Calculate statistics
                distances = [doc.distance for doc in documents if doc.distance is not None]
                content_lengths = [len(doc.content) for doc in documents]
                
                if distances:
                    analysis['distance_stats'] = {
                        'min': min(distances),
                        'max': max(distances),
                        'average': sum(distances) / len(distances)
                    }
                
                analysis['content_stats'] = {
                    'min_length': min(content_lengths),
                    'max_length': max(content_lengths),
                    'average_length': sum(content_lengths) / len(content_lengths)
                }
                
                # Add detailed results
                for doc in documents:
                    result = {
                        'id': doc.id,
                        'distance': doc.distance,
                        'content_length': len(doc.content),
                        'metadata': doc.metadata,
                        'content_preview': doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
                    }
                    analysis['results'].append(result)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Search and analysis failed: {str(e)}")
            return {
                'query': query,
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
    def analyze_metadata_patterns(self) -> Dict[str, Any]:
        """
        Analyze metadata patterns across all documents.
        
        Returns:
            Dict[str, Any]: Metadata pattern analysis
        """
        try:
            # Get all documents
            documents = self.chroma_browser.get_all_documents()
            
            analysis = {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_documents': len(documents),
                'metadata_analysis': {}
            }
            
            if not documents:
                return analysis
                
            # Analyze each metadata field
            metadata_fields = {}
            
            for doc in documents:
                for key, value in doc.metadata.items():
                    if key not in metadata_fields:
                        metadata_fields[key] = {
                            'values': {},
                            'total_occurrences': 0,
                            'data_types': set()
                        }
                    
                    # Count value occurrences
                    str_value = str(value)
                    if str_value not in metadata_fields[key]['values']:
                        metadata_fields[key]['values'][str_value] = 0
                    metadata_fields[key]['values'][str_value] += 1
                    metadata_fields[key]['total_occurrences'] += 1
                    
                    # Track data types
                    metadata_fields[key]['data_types'].add(type(value).__name__)
            
            # Process analysis results
            for field, data in metadata_fields.items():
                # Convert sets to lists for JSON serialization
                data['data_types'] = list(data['data_types'])
                
                # Calculate statistics
                unique_values = len(data['values'])
                most_common = max(data['values'].items(), key=lambda x: x[1])
                
                analysis['metadata_analysis'][field] = {
                    'unique_values': unique_values,
                    'total_occurrences': data['total_occurrences'],
                    'data_types': data['data_types'],
                    'most_common_value': {
                        'value': most_common[0],
                        'count': most_common[1],
                        'percentage': (most_common[1] / data['total_occurrences']) * 100
                    },
                    'coverage': (data['total_occurrences'] / len(documents)) * 100
                }
                
                # Add top values if there are many
                if unique_values > 5:
                    top_values = sorted(data['values'].items(), key=lambda x: x[1], reverse=True)[:5]
                    analysis['metadata_analysis'][field]['top_values'] = [
                        {'value': value, 'count': count} for value, count in top_values
                    ]
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Metadata analysis failed: {str(e)}")
            return {
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
    def export_analysis_report(
        self, 
        output_path: str,
        include_search_examples: bool = True
    ) -> None:
        """
        Export comprehensive analysis report.
        
        Args:
            output_path: Path to save the report
            include_search_examples: Whether to include search examples
        """
        try:
            # Generate comprehensive report
            report = {
                'report_type': 'comprehensive_memory_analysis',
                'generated_at': datetime.now().isoformat(),
                'database_info': {
                    'path': str(self.db_path),
                    'collection': self.chroma_browser.collection_name
                }
            }
            
            # Add memory summary
            report['memory_summary'] = self.get_memory_summary(
                include_samples=True,
                sample_count=3
            )
            
            # Add metadata analysis
            report['metadata_patterns'] = self.analyze_metadata_patterns()
            
            # Add search examples if requested
            if include_search_examples:
                example_queries = [
                    "incident management",
                    "procedures", 
                    "contact",
                    "emergency",
                    "escalation"
                ]
                
                search_examples = {}
                for query in example_queries:
                    try:
                        search_result = self.search_and_analyze(query, n_results=3)
                        if search_result.get('total_results', 0) > 0:
                            search_examples[query] = search_result
                    except Exception as e:
                        self.logger.warning(f"Search example failed for '{query}': {str(e)}")
                        
                if search_examples:
                    report['search_examples'] = search_examples
            
            # Save report
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Analysis report exported to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export analysis report: {str(e)}")
            raise
            
    def quick_stats(self) -> str:
        """
        Get quick statistics as formatted string.
        
        Returns:
            str: Formatted statistics
        """
        try:
            stats = self.chroma_browser.get_collection_stats()
            
            output = []
            output.append(f"📊 Memory Statistics")
            output.append(f"Collection: {stats.collection_name}")
            output.append(f"Documents: {stats.total_documents:,}")
            output.append(f"Metadata Fields: {len(stats.metadata_keys)}")
            
            if stats.content_stats:
                avg_length = stats.content_stats.get('average_length', 0)
                total_chars = stats.content_stats.get('total_characters', 0)
                output.append(f"Total Characters: {total_chars:,}")
                output.append(f"Average Length: {avg_length:.1f}")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"❌ Failed to get statistics: {str(e)}"