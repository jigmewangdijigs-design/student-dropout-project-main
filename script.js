// =========================================================
// COLORS
// =========================================================

const green = '#2c7a57';
const lime = '#b9e86b';
const ink = '#10251e';
const muted = '#dbe4dd';


// =========================================================
// CHART DEFAULT SETTINGS
// =========================================================

Chart.defaults.font.family = 'Manrope';
Chart.defaults.color = '#66766f';


// =========================================================
// CLASS DISTRIBUTION — DOUGHNUT CHART
// =========================================================

new Chart(document.getElementById('classChart'), {
    type: 'doughnut',

    data: {
        labels: [
            'Graduate',
            'Dropout',
            'Enrolled'
        ],

        datasets: [
            {
                data: [
                    2209,
                    1421,
                    794
                ],

                backgroundColor: [
                    lime,
                    '#57a77c',
                    '#dce5df'
                ],

                borderWidth: 0
            }
        ]
    },

    options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '72%',

        plugins: {
            legend: {
                position: 'right',

                labels: {
                    color: '#c5d2cc',
                    boxWidth: 8,
                    usePointStyle: true,

                    font: {
                        size: 9
                    }
                }
            }
        }
    }
});


// =========================================================
// CLASS DISTRIBUTION — BAR CHART
// =========================================================

new Chart(document.getElementById('distributionChart'), {
    type: 'bar',

    data: {
        labels: [
            'Graduate',
            'Dropout',
            'Enrolled'
        ],

        datasets: [
            {
                data: [
                    2209,
                    1421,
                    794
                ],

                backgroundColor: [
                    green,
                    '#75ad91',
                    '#bdd2c5'
                ],

                borderRadius: 8,
                borderSkipped: false
            }
        ]
    },

    options: {
        plugins: {
            legend: {
                display: false
            }
        },

        scales: {
            y: {
                beginAtZero: true,

                grid: {
                    color: '#edf0ec'
                }
            },

            x: {
                grid: {
                    display: false
                }
            }
        }
    }
});


// =========================================================
// MODEL PERFORMANCE CHART
// =========================================================

new Chart(document.getElementById('modelChart'), {
    type: 'bar',

    data: {
        labels: [
            'Baseline',
            'Logistic Reg.',
            'Decision Tree',
            'Random Forest',
            'Histogram Gradient Boosting'
        ],

        datasets: [
            {
                label: 'Accuracy %',

                data: [
                    49.93,
                    77.62,
                    68.01,
                    77.39,
                    77.62
                ],

                backgroundColor: green,
                borderRadius: 6
            },

            {
                label: 'Weighted F1 %',

                data: [
                    33.25,
                    76.38,
                    68.29,
                    75.30,
                    76.72
                ],

                backgroundColor: '#b9e86b',
                borderRadius: 6
            }
        ]
    },

    options: {
        indexAxis: 'y',

        plugins: {
            legend: {
                labels: {
                    boxWidth: 9,
                    usePointStyle: true,

                    font: {
                        size: 9
                    }
                }
            }
        },

        scales: {
            x: {
                min: 0,
                max: 100,

                grid: {
                    color: '#edf0ec'
                }
            },

            y: {
                grid: {
                    display: false
                }
            }
        }
    }
});


// =========================================================
// SCROLL REVEAL ANIMATION
// =========================================================

const observer = new IntersectionObserver(
    (entries) => {
        entries.forEach((entry) => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    },
    {
        threshold: 0.08
    }
);

document.querySelectorAll('.reveal').forEach((element) => {
    observer.observe(element);
});


// =========================================================
// MOBILE NAVIGATION MENU
// =========================================================

const btn = document.querySelector('.menu-btn');
const links = document.querySelector('.nav-links');

btn.addEventListener('click', () => {
    links.classList.toggle('open');
});


// Close mobile menu after clicking a navigation link
document.querySelectorAll('.nav-links a').forEach((link) => {
    link.addEventListener('click', () => {
        links.classList.remove('open');
    });
});
