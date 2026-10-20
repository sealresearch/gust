// Benchmark samples: real Mobile Actions evaluation rows (CC BY 4.0), each with its own tools; the page scores
// Relay's calls against the gold calls live (app.mjs runBench / drawBench).
export default {
  key: 'bench',
  custom: 'bench',
  label: 'Benchmark samples',
  subtitle: 'real Mobile Actions rows, scored live',
  tagline: '60 rows from Google\'s Mobile Actions eval set (CC BY 4.0), run in this tab and checked against the gold calls',
  placeholder: 'press Run to score the next 20 rows',
  tools: [],
  init: () => ({}),
  examples: [
    {title: 'Next 20 rows', sub: 'score live', q: 'next'},
  ],
  help: 'Press Run (or the chip) to score the next 20 Mobile Actions rows. Each row brings its own tools and date context.',
};
