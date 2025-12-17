document.addEventListener('DOMContentLoaded', () => {
    const grid = document.getElementById('experiment-grid');
    const lastUpdated = document.getElementById('last-updated');

    // Fetch Manifest
    fetch('manifest.json')
        .then(response => response.json())
        .then(data => {
            lastUpdated.textContent = `Last Updated: ${new Date(data.generated_at).toLocaleString()}`;
            renderExperiments(data.experiments);
        })
        .catch(err => {
            console.error('Error loading manifest:', err);
            grid.innerHTML = '<div class="error-state">Failed to load experiments. Run generate_manifest.py first.</div>';
        });

    function renderExperiments(experiments) {
        grid.innerHTML = ''; // Clear loading state

        if (experiments.length === 0) {
            grid.innerHTML = '<div class="empty-state">No experiments found.</div>';
            return;
        }

        const template = document.getElementById('card-template');

        experiments.forEach(exp => {
            const clone = template.content.cloneNode(true);
            
            // Populate Data
            clone.querySelector('.exp-id').textContent = formatExperimentName(exp.id);
            clone.querySelector('.dataset-badge').textContent = exp.dataset.toUpperCase();
            clone.querySelector('.time-badge').textContent = new Date(exp.timestamp).toLocaleDateString();
            clone.querySelector('.best-variator').textContent = formatVariatorName(exp.best_variator);

            // Setup View Button
            const btn = clone.querySelector('.view-btn');
            btn.addEventListener('click', () => {
                // Assuming server is run from ROOT, paths are: /results/...
                // If dashboard is at /dashboard/, we need ../results/...
                window.open(`../${exp.path}/summary.json`, '_blank');
            });

            // Populate Images
            // exp.path is "results/experiments/..."
            // We need to traverse up from /dashboard/ to root, then down to results
            const basePath = `../${exp.path}`;
            
            clone.querySelector('.chart-accuracy').src = `${basePath}/accuracy_comparison.png`;
            clone.querySelector('.chart-latency').src = `${basePath}/latency_distribution.png`;
            clone.querySelector('.chart-tokens').src = `${basePath}/token_usage.png`;

            // Append to Grid
            grid.appendChild(clone);
        });
    }

    function formatExperimentName(name) {
        // sentiment_2025... -> Sentiment (2025...)
        const parts = name.split('_');
        return `${parts[0].charAt(0).toUpperCase() + parts[0].slice(1)}`;
    }

    function formatVariatorName(name) {
        return name.replace('Variator', '').replace(/([A-Z])/g, ' $1').trim();
    }
});
