import React from 'react'
import { Twitter, Facebook, Instagram } from 'lucide-react';

const Footer = () => {
    return (
        <footer className='flex flex-col items-center justify-center mb-0 mt-20 sm:mt-24 md:mt-30 gap-6 sm:gap-8 px-4 py-8'>
            <div className='flex flex-col sm:flex-row text-white gap-4 sm:gap-8 md:gap-16 lg:gap-20 text-center sm:text-left'>
                <h1 className='cursor-pointer hover:text-[#8013ec] transition-colors text-sm sm:text-base'>Privacy Policy</h1>
                <h1 className='cursor-pointer hover:text-[#8013ec] transition-colors text-sm sm:text-base'>Terms of Service</h1>
                <h1 className='cursor-pointer hover:text-[#8013ec] transition-colors text-sm sm:text-base'>Contact Us</h1>
            </div>
            <div className='flex gap-5 sm:gap-6 md:gap-8 text-white'>
                <Twitter className='cursor-pointer hover:text-[#8013ec] transition-colors w-5 h-5 sm:w-6 sm:h-6' />
                <Facebook className='cursor-pointer hover:text-[#8013ec] transition-colors w-5 h-5 sm:w-6 sm:h-6' />
                <Instagram className='cursor-pointer hover:text-[#8013ec] transition-colors w-5 h-5 sm:w-6 sm:h-6' />
            </div>
            <div className='text-white text-xs sm:text-sm text-center px-4'>&copy; 2025 Sentiment Insights. All rights reserved.</div>
        </footer>
    )
}

export default Footer