"""HTML backtest report with Plotly interactive charts."""
import pandas as pd
import numpy as np
import os
from typing import Dict, Optional


class BacktestReport:
    def __init__(self, result, strategy_name: str = "", output_dir: str = "."):
        self.result = result
        self.strategy_name = strategy_name
        self.output_dir = output_dir

    def generate(self, include_attribution: Dict = None,
                 include_stress: pd.DataFrame = None) -> str:
        html = self._build_html(include_attribution, include_stress)
        fname = f"backtest_report_{self.strategy_name}_{pd.Timestamp.now().strftime('%Y%m%d')}.html"
        fpath = os.path.join(self.output_dir, fname)
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(html)
        return fpath

    def _build_html(self, attribution, stress) -> str:
        r = self.result
        metrics = [
            ('Strategy', self.strategy_name),
            ('Initial Capital', f"{r.initial_capital:,.0f}"),
            ('Final Capital', f"{r.final_capital:,.0f}"),
            ('Total Return', f"{r.total_return*100:.2f}%"),
            ('Annual Return', f"{r.annual_return*100:.2f}%"),
            ('Max Drawdown', f"{r.max_drawdown*100:.2f}%"),
            ('Sharpe Ratio', f"{r.sharpe_ratio:.2f}"),
            ('Total Trades', str(r.total_trades)),
            ('Win Rate', f"{r.win_rate:.1f}%"),
        ]
        metric_rows = "\n".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in metrics
        )
        equity_div = ""
        if r.daily_capital is not None and not r.daily_capital.empty:
            equity_records = r.daily_capital[['date', 'total_value']].copy()
            equity_records['date'] = equity_records['date'].astype(str)
            equity_json = equity_records.to_json(orient='records')
            equity_div = f"""
            <div id="equity_chart" style="width:100%;height:500px;"></div>
            <script>
            var equityData = {equity_json};
            var dates = equityData.map(function(d) {{ return d.date; }});
            var values = equityData.map(function(d) {{ return d.total_value; }});
            var trace = {{x: dates, y: values, type: 'scatter', name: 'Equity', line: {{color: '#1f77b4'}}}};
            var layout = {{title: 'Equity Curve', xaxis: {{title: 'Date'}}, yaxis: {{title: 'Total Value'}},
                          margin: {{t: 40, r: 20, b: 40, l: 60}}, hovermode: 'x unified'}};
            Plotly.newPlot('equity_chart', [trace], layout);
            </script>
            """
        stress_rows = ""
        if stress is not None and not stress.empty:
            stress_records = stress.to_dict(orient='records')
            stress_rows = "<h2>Stress Test Results</h2><table><tr><th>Scenario</th><th>Return</th><th>Max DD</th><th>Sharpe</th></tr>"
            for s in stress_records:
                ret_str = f"{s.get('total_return', 0)*100:.1f}%" if s.get('total_return') is not None else "N/A"
                dd_str = f"{s.get('max_drawdown', 0)*100:.1f}%" if s.get('max_drawdown') is not None else "N/A"
                sh_str = f"{s.get('sharpe_ratio', 0):.2f}" if s.get('sharpe_ratio') is not None else "N/A"
                stress_rows += f"<tr><td>{s.get('scenario','')}</td><td>{ret_str}</td><td>{dd_str}</td><td>{sh_str}</td></tr>"
            stress_rows += "</table>"
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Backtest Report - {self.strategy_name}</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
body {{ font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; margin: 40px; background: #f8f9fa; color: #212529; }}
.container {{ max-width: 960px; margin: 0 auto; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
h2 {{ color: #333; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
td, th {{ border: 1px solid #dee2e6; padding: 10px 16px; text-align: left; }}
th {{ background: #1f77b4; color: white; }}
tr:nth-child(even) {{ background: #f2f2f2; }}
.chart-container {{ background: white; padding: 20px; margin: 20px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-radius: 4px; }}
</style></head><body>
<div class="container">
<h1>Backtest Report: {self.strategy_name}</h1>
<h2>Performance Metrics</h2>
<table>{metric_rows}</table>
<div class="chart-container">{equity_div}</div>
{stress_rows}
</div></body></html>"""
