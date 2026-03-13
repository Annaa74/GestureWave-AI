// Initialize AOS Animation Library
document.addEventListener('DOMContentLoaded', () => {
    AOS.init({
        duration: 800,
        once: true,
        offset: 50,
        easing: 'ease-out-cubic'
    });

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // Advanced Parallax Effect for floating widgets
    document.addEventListener('mousemove', parallax);

    function parallax(e) {
        document.querySelectorAll('.parallax-item').forEach(function (move) {
            const moving_value = move.getAttribute('data-speed');
            const x = (e.clientX * moving_value) / 100;
            const y = (e.clientY * moving_value) / 100;

            move.style.transform = `translateX(${x}px) translateY(${y}px)`;
        });
    }

    // Scroll-based Navbar Styling
    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(255, 255, 255, 0.9)';
            navbar.style.boxShadow = '0 4px 20px rgba(0,0,0,0.05)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.7)';
            navbar.style.boxShadow = 'none';
        }
    });
});
