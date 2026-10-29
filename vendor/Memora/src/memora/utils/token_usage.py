# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

class TokenUsageCallback:
    def __init__(self):
        """
        Initialize the callback with token usage tracking and predefined cost per token.
        """
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.source_usage = {}  # {source: {model: {'prompt_tokens': x, 'completion_tokens': y, 'total_tokens': z}}}
        self.model_usage = {}  # {model: {'prompt_tokens': x, 'completion_tokens': y, 'total_tokens': z}}

        # Cost per model (per million tokens)
        self.cost_per_token = {
            "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.60 / 1e6},
            "gpt-4o": {"input": 2.50 / 1e6, "output": 10.00 / 1e6},
            "o1-mini": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
            "o1": {"input": 15.00 / 1e6, "output": 60.00 / 1e6},
            "o3-mini": {"input": 1.10 / 1e6, "output": 4.40 / 1e6},
            "gpt-4.1-mini": {"input": 0.4 / 1e6, "output": 1.60 / 1e6},
            "gpt-4.1": {"input": 2.0 / 1e6, "output": 8.00 / 1e6},
        }

    def update(self, prompt_tokens: int, completion_tokens: int, model: str, source: str = ""):
        """
        Update token usage for a given source and model.
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)

        if source not in self.source_usage:
            self.source_usage[source] = {}
        if model not in self.source_usage[source]:
            self.source_usage[source][model] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

        if model not in self.model_usage:
            self.model_usage[model] = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

        self.source_usage[source][model]['prompt_tokens'] += prompt_tokens
        self.source_usage[source][model]['completion_tokens'] += completion_tokens
        self.source_usage[source][model]['total_tokens'] += (prompt_tokens + completion_tokens)
        self.model_usage[model]['prompt_tokens'] += prompt_tokens
        self.model_usage[model]['completion_tokens'] += completion_tokens
        self.model_usage[model]['total_tokens'] += (prompt_tokens + completion_tokens)

    def token_usage_report(self) -> str:
        """
        Generate a report of token usage and cost and return it as a string.
        
        # Example Usage
            callback = TokenUsageCallback()
            callback.update(500, 200, model="gpt-4o", source="test1")
            callback.update(1000, 300, model="gpt-4o-mini", source="test1")
            callback.update(800, 400, model="o1", source="test2")

            # Get the report as a string
            report = callback.generate_usage_report()
            print(report)  # You can print, log, or save it to a file
        """
        total_cost = 0
        for model, usage in self.model_usage.items():
            if model in self.cost_per_token:
                total_cost += (usage['prompt_tokens'] * self.cost_per_token[model]["input"] +
                               usage['completion_tokens'] * self.cost_per_token[model]["output"])

        report_lines = []
        report_lines.append("===========================")
        report_lines.append("      TOKEN USAGE REPORT   ")
        report_lines.append("===========================")
        report_lines.append("Total Prompt Tokens     : " + str(self.prompt_tokens))
        report_lines.append("Total Completion Tokens : " + str(self.completion_tokens))
        report_lines.append("Total Tokens            : " + str(self.total_tokens))
        report_lines.append("Total Estimated Cost    : $" + format(total_cost, ".6f"))
        report_lines.append("\n==============================")
        report_lines.append("   BREAKDOWN BY SOURCE/MODEL  ")
        report_lines.append("==============================")

        source_costs = {}
        for source, models in self.source_usage.items():
            report_lines.append("\nSource: " + source)
            total_source_cost = 0
            for model, usage in models.items():
                model_cost = 0
                if model in self.cost_per_token:
                    model_cost = (usage['prompt_tokens'] * self.cost_per_token[model]["input"] +
                                  usage['completion_tokens'] * self.cost_per_token[model]["output"])
                total_source_cost += model_cost
                report_lines.append("  - Model: " + model)
                report_lines.append("    Prompt Tokens    : " + str(usage['prompt_tokens']))
                report_lines.append("    Completion Tokens: " + str(usage['completion_tokens']))
                report_lines.append("    Total Tokens     : " + str(usage['total_tokens']))
                report_lines.append("    Cost             : $" + format(model_cost, ".6f"))
            source_costs[source] = total_source_cost
            report_lines.append("  >> Total Cost for Source: $" + format(total_source_cost, ".6f"))
            report_lines.append("--------------------------------")

        report_lines.append("\n===========================")
        report_lines.append("   COST BREAKDOWN BY MODEL")
        report_lines.append("===========================")
        for model, usage in self.model_usage.items():
            if model in self.cost_per_token:
                model_cost = (usage['prompt_tokens'] * self.cost_per_token[model]["input"] +
                              usage['completion_tokens'] * self.cost_per_token[model]["output"])
                report_lines.append("Model: " + model)
                report_lines.append("  Prompt Tokens    : " + str(usage['prompt_tokens']))
                report_lines.append("  Completion Tokens: " + str(usage['completion_tokens']))
                report_lines.append("  Total Tokens     : " + str(usage['total_tokens']))
                report_lines.append("  Total Cost       : $" + format(model_cost, ".6f"))
                report_lines.append("---------------------------")

        return "\n".join(report_lines)
