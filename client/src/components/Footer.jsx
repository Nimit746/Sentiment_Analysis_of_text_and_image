import React from 'react'
import { Twitter, Facebook, Instagram } from 'lucide-react';
const Footer = () => {
    return (
        <footer className='flex flex-col items-center justify-center mb-0 mt-30 gap-8'>
            <div className='flex text-white gap-70 '>
                <h1>Privacy Policy</h1>
                <h1>Terms of Service</h1>
                <h1>Contact Us</h1>
            </div>
            <div className='flex gap-5 text-white'>
                <Twitter />
                <Facebook />
                <Instagram/>
            </div>
            <div className='text-white'> &copy; 2025 Sentiment ingsights. All rights reserved.</div>
        </footer>
    )
}

export default Footer