/* eslint-disable no-unused-vars */
import React from 'react';
import { Home, Search, ArrowLeft, Mail } from 'lucide-react';

const QuickLink = ({ Icon, title, description, onClick }) => (
    <button
        onClick={onClick}
        className="bg-gray-800/30 backdrop-blur-sm border border-gray-700 rounded-xl p-6 hover:border-purple-500 transition-all duration-300 group text-left w-full"
    >
        <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-lg bg-purple-500/20 flex items-center justify-center flex-shrink-0 group-hover:bg-purple-500/30 transition-colors">
                <Icon size={24} className="text-purple-400" />
            </div>
            <div>
                <h3 className="text-lg font-semibold text-white mb-1 group-hover:text-purple-400 transition-colors">
                    {title}
                </h3>
                <p className="text-gray-400 text-sm">{description}</p>
            </div>
        </div>
    </button>
);

export default function NotFound() {
    const handleGoHome = () => {
        console.log('Navigate to home');
        // Add your navigation logic here
        // Example: window.location.href = '/';
    };

    const handleGoBack = () => {
        console.log('Navigate back');
        window.history.back();
    };

    const handleSearch = () => {
        console.log('Open search');
        // Add your search logic here
    };

    const handleContact = () => {
        console.log('Navigate to contact');
        // Add your navigation logic here
        // Example: window.location.href = '/contact';
    };

    const quickLinks = [
        {
            Icon: Home,
            title: 'Go to Homepage',
            description: 'Return to the main page and explore our features',
            onClick: handleGoHome
        },
        {
            Icon: Search,
            title: 'Search',
            description: 'Find what you\'re looking for using our search',
            onClick: handleSearch
        },
        {
            Icon: Mail,
            title: 'Contact Support',
            description: 'Get help from our support team',
            onClick: handleContact
        }
    ];

    return (
        <div className="min-h-screen bg-linear-to-br from-gray-900 via-purple-900 to-gray-900 flex items-center justify-center px-4 py-12">
            <div className="max-w-2xl w-full text-center">
                {/* 404 Illustration */}
                <div className="mb-8 relative">
                    <div className="text-[150px] md:text-[200px] font-bold text-transparent bg-clip-text bg-linear-to-r from-purple-400 to-blue-400 leading-none select-none">
                        404
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-32 h-32 md:w-40 md:h-40 rounded-full bg-purple-500/10 blur-3xl"></div>
                    </div>
                </div>

                {/* Error Message */}
                <div className="mb-12">
                    <h1 className="text-3xl md:text-4xl font-bold text-white mb-4">
                        Page Not Found
                    </h1>
                    <p className="text-gray-300 text-lg leading-relaxed max-w-md mx-auto">
                        Oops! The page you're looking for seems to have wandered off into the digital void.
                        Don't worry, we'll help you find your way back.
                    </p>
                </div>

                {/* Quick Actions */}
                <div className="flex gap-4 justify-center mb-12 flex-wrap">
                    <button
                        onClick={handleGoBack}
                        className="bg-purple-500 hover:bg-purple-600 text-white font-medium px-6 py-3 rounded-lg transition-colors duration-200 shadow-lg hover:shadow-xl active:scale-[0.98] flex items-center gap-2"
                    >
                        <ArrowLeft size={20} />
                        Go Back
                    </button>
                    <button
                        onClick={handleGoHome}
                        className="bg-gray-800 hover:bg-gray-700 text-white font-medium px-6 py-3 rounded-lg border border-gray-600 transition-colors duration-200 active:scale-[0.98] flex items-center gap-2"
                    >
                        <Home size={20} />
                        Home
                    </button>
                </div>

                {/* Quick Links */}
                <div className="space-y-4">
                    <h2 className="text-xl font-semibold text-white mb-6">
                        Quick Links
                    </h2>
                    <div className="grid md:grid-cols-3 gap-4">
                        {quickLinks.map(link => (
                            <QuickLink key={link.title} {...link} />
                        ))}
                    </div>
                </div>

                {/* Additional Help */}
                <div className="mt-12 bg-gray-800/20 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
                    <p className="text-gray-400 text-sm">
                        Still can't find what you're looking for?
                        <button
                            onClick={handleContact}
                            className="text-purple-400 hover:text-purple-300 transition-colors ml-1 underline"
                        >
                            Contact our support team
                        </button>
                    </p>
                </div>
            </div>
        </div>
    );
}