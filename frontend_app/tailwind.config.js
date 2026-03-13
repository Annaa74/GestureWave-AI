/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                background: '#fafafb',
                surface: '#ffffff',
                primary: '#2563eb',
                secondary: '#8b5cf6',
                text: {
                    main: '#1a1a24',
                    muted: '#5c5d71'
                }
            },
            fontFamily: {
                sans: ['Inter', 'sans-serif'],
                display: ['Outfit', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
