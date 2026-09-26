document.getElementById('analyze-btn').addEventListener('click', async () => {
    const text = document.getElementById('text-input').value;
    if (text.length < 20) return alert("Please enter at least 20 characters of text to analyze.");
    
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    
    try {
        const response = await fetch('/analyze', { 
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        let html = `<h2>Verdict: ${data.verdict}</h2>`;
        html += `<p><strong>AI Probability:</strong> ${(data.ai_probability * 100).toFixed(2)}%</p>`;
        
        html += `<h3>Signal Breakdown:</h3><ul>`;
        html += `<li>Perplexity: ${(data.signal_breakdown.perplexity_engine * 100).toFixed(1)}%</li>`;
        html += `<li>Stylometric: ${(data.signal_breakdown.stylometric_engine * 100).toFixed(1)}%</li>`;
        html += `<li>Learned Classifier: ${(data.signal_breakdown.learned_classifier * 100).toFixed(1)}%</li>`;
        html += `</ul>`;
        
        if (data.evidence && data.evidence.length > 0) {
            html += `<h3>Top AI-Likely Sentences:</h3><ul>`;
            data.evidence.forEach(ev => {
                html += `<li>"${ev.sentence}" <em>(${(ev.ai_likelihood * 100).toFixed(1)}%)</em></li>`;
            });
            html += `</ul>`;
        }

        document.getElementById('results').innerHTML = html;
        document.getElementById('results').classList.remove('hidden');
    } catch (error) {
        alert("Error connecting to the Spider-Sense API. Is the server running?");
        console.error(error);
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
});