"""
LLM Report Generator for EKYC PAD Analysis Dashboard
=====================================================

Generates automated reports using free LLM APIs:
- Ollama (local, free, no rate limits)
- Groq (free tier, very fast)
- Hugging Face (free inference)

Usage:
    python llm_report_generator.py --api ollama --analysis_data results.json
"""

import json
import requests
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class LLMReportGenerator:
    """Generate analysis reports using free LLM APIs."""

    def __init__(self, api_type='ollama', model_name='mistral'):
        """
        Initialize LLM reporter.
        
        Args:
            api_type: 'ollama', 'groq', or 'huggingface'
            model_name: model to use
        """
        self.api_type = api_type
        self.model_name = model_name
        
        if api_type == 'ollama':
            self.endpoint = 'http://localhost:11434/api/generate'
        elif api_type == 'groq':
            self.api_key = os.getenv('GROQ_API_KEY', '')  # Get from env
            self.endpoint = 'https://api.groq.com/openai/v1/chat/completions'
        elif api_type == 'huggingface':
            self.api_key = os.getenv('HF_API_KEY', '')
            self.endpoint = f'https://api-inference.huggingface.co/models/{model_name}'
    
    def generate_analysis_summary(self, results: Dict) -> str:
        """Generate executive summary from detection results."""
        
        prompt = f"""
Analyze the following EKYC Presentation Attack Detection (PAD) results and provide a brief executive summary:

Detection Results:
- Total videos analyzed: {results.get('total_videos', 0)}
- Real samples detected: {results.get('real_count', 0)}
- Spoofing samples detected: {results.get('spoof_count', 0)}
- Overall accuracy: {results.get('accuracy', 0):.2%}

Attack types detected:
{json.dumps(results.get('attack_breakdown', {}), indent=2)}

Model confidence ranges:
- Min: {results.get('confidence_min', 0):.2%}
- Max: {results.get('confidence_max', 0):.2%}
- Mean: {results.get('confidence_mean', 0):.2%}

Temporal consistency metrics:
- Average: {results.get('temporal_avg', 0):.3f}
- Std Dev: {results.get('temporal_std', 0):.3f}

Provide a 3-4 sentence executive summary highlighting key findings and recommendations.
Keep it concise and technical.
"""
        
        return self._call_llm(prompt)
    
    def generate_explainability_report(self, sample: Dict) -> str:
        """Generate explainability report for a single sample."""
        
        prompt = f"""
Explain the following EKYC PAD (Presentation Attack Detection) prediction in technical detail:

Sample Details:
- Predicted Label: {sample.get('label', 'UNKNOWN')}
- Confidence: {sample.get('confidence', 0):.2%}
- Attack Type: {sample.get('attack_type', 'N/A')}
- Temporal Consistency Score: {sample.get('temporal_score', 0):.3f}

Model Features:
- CLS Token Activation: [complex vector features]
- Attention Patterns: Detected motion in {sample.get('motion_regions', 0)} regions

Explain:
1. Why the model made this prediction (2-3 sentences)
2. Key indicators used for spoofing detection (list 3-4)
3. Confidence assessment (high/medium/low and why)
4. Recommendations for borderline cases

Format as clear bullet points.
"""
        
        return self._call_llm(prompt)
    
    def generate_attack_analysis(self, attack_breakdown: Dict) -> str:
        """Analyze distribution and patterns of detected attacks."""
        
        prompt = f"""
Analyze the following distribution of presentation attack types detected in EKYC verification:

Attack Distribution:
{json.dumps(attack_breakdown, indent=2)}

Provide:
1. Summary of attack patterns (which attacks are most common)
2. Potential vulnerabilities (which types are hardest to detect)
3. Recommendations for improving detection
4. Risk assessment for deployment

Keep technical but accessible.
"""
        
        return self._call_llm(prompt)
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API based on configured type."""
        
        try:
            if self.api_type == 'ollama':
                return self._call_ollama(prompt)
            elif self.api_type == 'groq':
                return self._call_groq(prompt)
            elif self.api_type == 'huggingface':
                return self._call_huggingface(prompt)
        except Exception as e:
            return f"[Error generating report: {str(e)}]"
    
    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama API."""
        
        try:
            response = requests.post(
                self.endpoint,
                json={
                    'model': self.model_name,
                    'prompt': prompt,
                    'stream': False,
                    'temperature': 0.7
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['response'].strip()
            else:
                return f"[Ollama error: {response.status_code}]"
        
        except requests.exceptions.ConnectionError:
            return "[Ollama not running. Start with: ollama serve]"
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API (free tier)."""
        
        try:
            response = requests.post(
                self.endpoint,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'mixtral-8x7b-32768',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.7,
                    'max_tokens': 1024
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"[Groq error: {response.status_code}]"
        
        except Exception as e:
            return f"[Groq API error: {str(e)}]"
    
    def _call_huggingface(self, prompt: str) -> str:
        """Call Hugging Face Inference API."""
        
        try:
            response = requests.post(
                self.endpoint,
                headers={'Authorization': f'Bearer {self.api_key}'},
                json={'inputs': prompt},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '[Empty response]')
                return str(result)
            else:
                return f"[HF error: {response.status_code}]"
        
        except Exception as e:
            return f"[HF API error: {str(e)}]"

class AnalysisDashboardGenerator:
    """Generate interactive HTML dashboard from analysis results."""
    
    def __init__(self, llm_reporter: LLMReportGenerator):
        self.llm = llm_reporter
    
    def generate_dashboard(self, results: Dict, output_path='dashboard.html') -> str:
        """Generate complete HTML dashboard."""
        
        # Generate LLM reports
        summary = self.llm.generate_analysis_summary(results)
        attack_analysis = self.llm.generate_attack_analysis(results.get('attack_breakdown', {}))
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EKYC PAD Analysis Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f7fa; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        header p {{ font-size: 1.1em; opacity: 0.9; }}
        
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .card h3 {{ color: #333; margin-bottom: 15px; font-size: 1.1em; }}
        .card .metric {{ font-size: 2.5em; font-weight: bold; color: #667eea; }}
        .card .label {{ color: #666; font-size: 0.9em; margin-top: 10px; }}
        
        .section {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px; }}
        .section h2 {{ color: #333; margin-bottom: 20px; border-bottom: 3px solid #667eea; padding-bottom: 10px; }}
        
        .report-text {{ color: #555; line-height: 1.8; font-size: 1em; margin-bottom: 15px; white-space: pre-wrap; }}
        
        .chart-container {{ position: relative; height: 400px; margin-bottom: 20px; }}
        
        .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; }}
        .stat-box {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }}
        .stat-box .value {{ font-size: 1.8em; font-weight: bold; color: #667eea; }}
        .stat-box .label {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
        
        .attack-types {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px; }}
        .attack-badge {{ background: #e0e7ff; color: #667eea; padding: 8px 15px; border-radius: 20px; font-weight: 500; }}
        
        footer {{ text-align: center; color: #999; margin-top: 40px; padding: 20px; border-top: 1px solid #ddd; }}
        
        @media (max-width: 768px) {{
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔐 EKYC PAD Analysis Dashboard</h1>
            <p>Presentation Attack Detection (Spoofing) Analysis Report</p>
            <p style="font-size: 0.9em; margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <!-- Key Metrics -->
        <div class="grid">
            <div class="card">
                <h3>Total Videos</h3>
                <div class="metric">{results.get('total_videos', 0)}</div>
                <div class="label">Analyzed</div>
            </div>
            <div class="card">
                <h3>Real Samples</h3>
                <div class="metric">{results.get('real_count', 0)}</div>
                <div class="label">Genuine Attempts</div>
            </div>
            <div class="card">
                <h3>Spoofing Detected</h3>
                <div class="metric">{results.get('spoof_count', 0)}</div>
                <div class="label">Attack Attempts</div>
            </div>
            <div class="card">
                <h3>Model Accuracy</h3>
                <div class="metric">{results.get('accuracy', 0):.1%}</div>
                <div class="label">Overall Performance</div>
            </div>
        </div>
        
        <!-- Executive Summary (LLM Generated) -->
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="report-text">{summary}</div>
        </div>
        
        <!-- Attack Distribution Chart -->
        <div class="section">
            <h2>🎯 Attack Type Distribution</h2>
            <div class="chart-container">
                <canvas id="attackChart"></canvas>
            </div>
            <div class="report-text">{attack_analysis}</div>
        </div>
        
        <!-- Detailed Statistics -->
        <div class="section">
            <h2>📈 Detailed Statistics</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="value">{results.get('confidence_mean', 0):.1%}</div>
                    <div class="label">Avg Confidence</div>
                </div>
                <div class="stat-box">
                    <div class="value">{results.get('temporal_avg', 0):.3f}</div>
                    <div class="label">Temporal Score</div>
                </div>
                <div class="stat-box">
                    <div class="value">{results.get('precision', 0):.1%}</div>
                    <div class="label">Precision</div>
                </div>
                <div class="stat-box">
                    <div class="value">{results.get('recall', 0):.1%}</div>
                    <div class="label">Recall</div>
                </div>
            </div>
        </div>
        
        <!-- Recommendations -->
        <div class="section">
            <h2>💡 Recommendations</h2>
            <ul style="margin-left: 20px; line-height: 1.8; color: #555;">
                <li>Deploy model with high confidence threshold (>0.95) for production</li>
                <li>Implement continuous monitoring of temporal consistency scores</li>
                <li>Retrain quarterly with new attack patterns</li>
                <li>Use ensemble with multiple face liveness detection methods</li>
            </ul>
        </div>
        
        <footer>
            <p>EKYC Presentation Attack Detection System v1.0</p>
            <p>For questions or concerns, contact the development team.</p>
        </footer>
    </div>
    
    <script>
        const attackData = {json.dumps(results.get('attack_breakdown', {}))};
        
        const ctx = document.getElementById('attackChart').getContext('2d');
        new Chart(ctx, {{
            type: 'doughnut',
            data: {{
                labels: Object.keys(attackData),
                datasets: [{{
                    data: Object.values(attackData),
                    backgroundColor: [
                        '#667eea',
                        '#764ba2',
                        '#f093fb',
                        '#4facfe'
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom' }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate EKYC PAD Analysis Reports')
    parser.add_argument('--api', choices=['ollama', 'groq', 'huggingface'], default='ollama',
                       help='LLM API to use')
    parser.add_argument('--analysis_data', type=str, default='analysis_results.json',
                       help='JSON file with analysis results')
    parser.add_argument('--output', type=str, default='dashboard.html',
                       help='Output HTML dashboard path')
    
    args = parser.parse_args()
    
    # Load analysis results
    if Path(args.analysis_data).exists():
        with open(args.analysis_data) as f:
            results = json.load(f)
    else:
        # Demo data
        results = {
            'total_videos': 100,
            'real_count': 60,
            'spoof_count': 40,
            'accuracy': 0.92,
            'precision': 0.94,
            'recall': 0.90,
            'confidence_mean': 0.876,
            'confidence_min': 0.51,
            'confidence_max': 0.99,
            'temporal_avg': 0.823,
            'temporal_std': 0.134,
            'attack_breakdown': {
                'VideoReplay': 15,
                'PrintedPhoto': 18,
                'Masked': 7
            }
        }
    
    # Generate report
    print(f"Generating report using {args.api} API...")
    reporter = LLMReportGenerator(api_type=args.api)
    dashboard_gen = AnalysisDashboardGenerator(reporter)
    
    output_file = dashboard_gen.generate_dashboard(results, args.output)
    print(f"✅ Dashboard generated: {output_file}")

if __name__ == '__main__':
    import os
    main()