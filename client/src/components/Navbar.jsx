import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Menu, X } from 'lucide-react'

const Navbar = () => {
    const [isMenuOpen, setIsMenuOpen] = useState(false);
    const navigate = useNavigate();

    const toggleMenu = () => {
        setIsMenuOpen(!isMenuOpen);
    };

    const handleNavClick = (path) => {
        setIsMenuOpen(false);
        navigate(path);
    };

    return (
        <nav className="flex items-center justify-between border-b border-solid border-b-[#362348] px-4 sm:px-6 lg:px-10 py-3 bg-[#1a1122] relative">
            {/* Logo Section */}
            <div className="flex items-center gap-2 sm:gap-3 text-white z-50">
                <div className="size-8 sm:size-10 transition-transform hover:scale-110">
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
                <h2 className="text-white text-sm sm:text-base lg:text-lg font-bold leading-tight tracking-[-0.015em]">
                    Sentiment Analyzer
                </h2>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden lg:flex flex-1 justify-end gap-4 xl:gap-8">
                <div className="flex items-center gap-6 xl:gap-9">
                    <Link className="text-white text-sm font-medium leading-normal hover:text-[#8013ec] transition-colors" to="/">Home</Link>
                    <Link className="text-white text-sm font-medium leading-normal hover:text-[#8013ec] transition-colors" to="/about">About</Link>
                    <Link className="text-white text-sm font-medium leading-normal hover:text-[#8013ec] transition-colors" to="/contact">Contact</Link>
                    <Link className="text-white text-sm font-medium leading-normal hover:text-[#8013ec] transition-colors" to="/sentiment-analysis">Try Out</Link>
                </div>
                <button
                    className="flex min-w-[84px] max-w-[480px] cursor-pointer items-center justify-center overflow-hidden rounded-xl h-10 px-4 bg-[#362348] text-white text-sm font-bold leading-normal tracking-[0.015em] hover:bg-[#4a2f5c] transition-all hover:scale-105"
                    onClick={() => navigate("/signup")}
                >
                    <span className="truncate">Sign Up</span>
                </button>
            </div>

            {/* Mobile Menu Button & Sign Up */}
            <div className="flex lg:hidden items-center gap-3 z-50">
                <button
                    className="flex min-w-[70px] cursor-pointer items-center justify-center overflow-hidden rounded-xl h-9 px-3 bg-[#362348] text-white text-xs font-bold leading-normal tracking-[0.015em] hover:bg-[#4a2f5c] transition-all"
                    onClick={() => navigate("/signup")}
                >
                    <span className="truncate">Sign Up</span>
                </button>
                <button
                    onClick={toggleMenu}
                    className="text-white p-2 hover:bg-[#362348] rounded-lg transition-colors"
                    aria-label="Toggle menu"
                >
                    {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
                </button>
            </div>

            {/* Mobile Menu Overlay */}
            {isMenuOpen && (
                <div className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden" onClick={toggleMenu}></div>
            )}

            {/* Mobile Navigation Menu */}
            <div className={`fixed top-[57px] right-0 w-64 h-[calc(100vh-57px)] bg-[#1a1122] border-l border-[#362348] transform transition-transform duration-300 ease-in-out z-40 lg:hidden ${isMenuOpen ? 'translate-x-0' : 'translate-x-full'
                }`}>
                <div className="flex flex-col p-6 gap-6">
                    <Link
                        className="text-white text-base font-medium leading-normal hover:text-[#8013ec] transition-colors py-2"
                        to="/"
                        onClick={() => handleNavClick('/')}
                    >
                        Home
                    </Link>
                    <Link
                        className="text-white text-base font-medium leading-normal hover:text-[#8013ec] transition-colors py-2"
                        to="/about"
                        onClick={() => handleNavClick('/about')}
                    >
                        About
                    </Link>
                    <Link
                        className="text-white text-base font-medium leading-normal hover:text-[#8013ec] transition-colors py-2"
                        to="/contact"
                        onClick={() => handleNavClick('/contact')}
                    >
                        Contact
                    </Link>
                    <Link
                        className="text-white text-base font-medium leading-normal hover:text-[#8013ec] transition-colors py-2"
                        to="/sentiment-analysis"
                        onClick={() => handleNavClick('/sentiment-analysis')}
                    >
                        Try Out
                    </Link>
                </div>
            </div>
        </nav>
    )
}

export default Navbar