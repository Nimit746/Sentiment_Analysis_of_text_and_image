/* eslint-disable no-unused-vars */
import React, { useState } from 'react';
import { Mail, Twitter, Facebook, Instagram } from 'lucide-react';

const InputField = ({ label, type = 'text', name, value, onChange, placeholder, rows }) => {
    const Component = rows ? 'textarea' : 'input';

    return (
        <div>
            <label htmlFor={name} className="block text-white text-sm font-medium mb-2">
                {label}
            </label>
            <Component
                id={name}
                type={type}
                name={name}
                value={value}
                onChange={onChange}
                placeholder={placeholder}
                rows={rows}
                className="w-full px-4 py-3 bg-gray-800/50 border border-gray-700 rounded-lg text-gray-300 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all resize-none"
            />
        </div>
    );
};

const SocialIcon = ({ Icon, label, link }) => (
    <button
        className="flex flex-col items-center gap-2 text-gray-400 hover:text-purple-400 transition-colors group"
        aria-label={label}
    >
        <a href={link} target='_blank'>
            <div className="w-10 h-10 rounded-full bg-gray-800/50 border border-gray-700 flex items-center justify-center group-hover:border-purple-500 transition-colors">
                <Icon size={18} />
            </div>
            <span className="text-xs">{label}</span>
        </a>
    </button>
);

export default function Contact() {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        subject: '',
        message: ''
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleSubmit = (e) => {
        e.preventDefault();

        if (!formData.name || !formData.email || !formData.subject || !formData.message) {
            alert('Please fill in all fields');
            return;
        }

        console.log('Form submitted:', formData);
        alert('Thank you for your message! We will get back to you soon.');
        setFormData({ name: '', email: '', subject: '', message: '' });
    };

    const formFields = [
        { label: 'Your Name', name: 'name', placeholder: 'Enter your name' },
        { label: 'Your Email', name: 'email', type: 'email', placeholder: 'Enter your email' },
        { label: 'Subject', name: 'subject', placeholder: 'Enter the subject' },
        { label: 'Message', name: 'message', placeholder: 'Enter your message', rows: 5 }
    ];

    const socialLinks = [
        { Icon: Twitter, label: 'Twitter' },
        { Icon: Facebook, label: 'Facebook' },
        { Icon: Instagram, label: 'Instagram' }
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 flex items-center justify-center px-4 py-12">
            <div className="max-w-md w-full">
                {/* Header */}
                <header className="text-center mb-8">
                    <h1 className="text-3xl font-bold text-white mb-4">Contact Us</h1>
                    <p className="text-gray-300 text-sm leading-relaxed">
                        We're here to help! Reach out to us with any questions, feedback, or support inquiries.
                        Our team is dedicated to providing prompt and helpful assistance.
                    </p>
                </header>

                {/* Contact Form */}
                <div className="space-y-5 mb-10">
                    {formFields.map(field => (
                        <InputField
                            key={field.name}
                            {...field}
                            value={formData[field.name]}
                            onChange={handleChange}
                        />
                    ))}

                    <button
                        onClick={handleSubmit}
                        className="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-3 rounded-lg transition-colors duration-200 shadow-lg hover:shadow-xl active:scale-[0.98]"
                    >
                        Submit
                    </button>
                </div>

                {/* Contact Information */}
                <footer className="text-center">
                    <h2 className="text-white text-lg font-semibold mb-4">Contact Information</h2>

                    <a
                        href="mailto:support@sentimentanalyzer.com"
                        className="flex items-center justify-center gap-2 text-gray-300 hover:text-purple-400 transition-colors mb-4"
                    >
                        <Mail size={18} />
                        <span>support@sentimentanalyzer.com</span>
                    </a>

                    <p className="text-gray-300 text-sm mb-4">Follow us on social media:</p>
                    <div className="flex justify-center gap-6">
                        {socialLinks.map(social => (
                            <SocialIcon key={social.label} {...social} />
                        ))}
                    </div>
                </footer>
            </div>
        </div>
    );
}