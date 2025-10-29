import React from 'react'
import { Link, useNavigate } from 'react-router-dom'

const Navbar = () => {

    const navigate = useNavigate();
    return (
        <nav className="flex items-center justify-between whitespace-nowrap border-b border-solid border-b-[#362348] px-4 sm:px-10 py-3 transition-al bg-[#1a1122]">
            <div className="flex items-center gap-3 sm:gap-4 text-white">
                <div className="size-4 transition-transform hover:scale-110">
                    <svg viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path
                            fillRule="evenodd"
                            clipRule="evenodd"
                            d="M24 18.4228L42 11.475V34.3663C42 34.7796 41.7457 35.1504 41.3601 35.2992L24 42V18.4228Z"
                            fill="currentColor"
                        />
                        <path
                            fillRule="evenodd"
                            clipRule="evenodd"
                            d="M24 8.18819L33.4123 11.574L24 15.2071L14.5877 11.574L24 8.18819ZM9 15.8487L21 20.4805V37.6263L9 32.9945V15.8487ZM27 37.6263V20.4805L39 15.8487V32.9945L27 37.6263ZM25.354 2.29885C24.4788 1.98402 23.5212 1.98402 22.646 2.29885L4.98454 8.65208C3.7939 9.08038 3 10.2097 3 11.475V34.3663C3 36.0196 4.01719 37.5026 5.55962 38.098L22.9197 44.7987C23.6149 45.0671 24.3851 45.0671 25.0803 44.7987L42.4404 38.098C43.9828 37.5026 45 36.0196 45 34.3663V11.475C45 10.2097 44.2061 9.08038 43.0155 8.65208L25.354 2.29885Z"
                            fill="currentColor"
                        />
                    </svg>
                </div>
                <h2 className="text-white text-base sm:text-lg font-bold leading-tight tracking-[-0.015em]">Sentiment Analyzer</h2>
            </div>
            <div className="flex flex-1 justify-end gap-4 sm:gap-8">
                <div className="hidden md:flex items-center gap-6 lg:gap-9">
                    <Link className="text-white text-sm font-medium leading-normal hover:text-[#8013ec] transition-colors" to="/">Home</Link>
                    <Link className="text-white text-sm font-medium leading-normal hover:text-[#8013ec] transition-colors" to="/about">About</Link>
                    <Link className="text-white text-sm font-medium leading-normal hover:text-[#8013ec] transition-colors" to="/contact">Contact</Link>
                    <Link className="text-white text-sm font-medium leading-normal hover:text-[#8013ec] transition-colors" to="/sentiment-analysis">Try Out</Link>
                </div>
                <button className="flex min-w-[70px] sm:min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-xl h-9 sm:h-10 px-3 sm:px-4 bg-[#362348] text-white text-xs sm:text-sm font-bold leading-normal tracking-[0.015em] hover:bg-[#4a2f5c] transition-all hover:scale-105" onClick={() => navigate("/signup")}>
                    <span className="truncate">Sign Up</span>
                </button>
            </div>
        </nav>
    )
}

export default Navbar
