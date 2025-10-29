/* eslint-disable no-unused-vars */
import React from 'react';
import { Target, Users, Zap, Shield, Award, TrendingUp } from 'lucide-react';
import { useNavigate } from 'react-router-dom'

const FeatureCard = ({ Icon, title, description }) => (
    <div className="bg-gray-800/30 backdrop-blur-sm border border-gray-700 rounded-xl p-6 hover:border-purple-500 transition-all duration-300 group">
        <div className="w-12 h-12 rounded-lg bg-purple-500/20 flex items-center justify-center mb-4 group-hover:bg-purple-500/30 transition-colors">
            <Icon size={24} className="text-purple-400" />
        </div>
        <h3 className="text-xl font-semibold text-white mb-2">{title}</h3>
        <p className="text-gray-400 text-sm leading-relaxed">{description}</p>
    </div>
);

const StatCard = ({ number, label }) => (
    <div className="text-center">
        <div className="text-4xl font-bold text-purple-400 mb-2">{number}</div>
        <div className="text-gray-400 text-sm">{label}</div>
    </div>
);

const TeamMember = ({ name, role, description }) => (
    <div className="bg-gray-800/30 backdrop-blur-sm border border-gray-700 rounded-xl p-6 hover:border-purple-500 transition-all duration-300">
        <div className="w-16 h-16 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 mb-4 flex items-center justify-center">
            <span className="text-2xl font-bold text-white">{name.charAt(0)}</span>
        </div>
        <h3 className="text-lg font-semibold text-white mb-1">{name}</h3>
        <p className="text-purple-400 text-sm mb-3">{role}</p>
        <p className="text-gray-400 text-sm leading-relaxed">{description}</p>
    </div>
);

export default function About() {

    const navigation = useNavigate();

    const features = [
        {
            Icon: Target,
            title: 'Our Mission',
            description: 'To empower businesses and individuals with advanced sentiment analysis tools that provide actionable insights from text data.'
        },
        {
            Icon: Zap,
            title: 'Fast & Accurate',
            description: 'Leveraging cutting-edge AI technology to deliver real-time sentiment analysis with industry-leading accuracy rates.'
        },
        {
            Icon: Shield,
            title: 'Secure & Private',
            description: 'Your data security is our priority. We employ enterprise-grade encryption and follow strict privacy protocols.'
        },
        {
            Icon: Users,
            title: 'User-Centric',
            description: 'Designed with simplicity in mind, our platform is intuitive and accessible for users of all technical backgrounds.'
        },
        {
            Icon: Award,
            title: 'Industry Leader',
            description: 'Recognized for excellence in NLP and sentiment analysis, trusted by Fortune 500 companies worldwide.'
        },
        {
            Icon: TrendingUp,
            title: 'Continuous Innovation',
            description: 'Constantly evolving with the latest advancements in machine learning and natural language processing.'
        }
    ];

    const stats = [
        { number: '1M+', label: 'Analyses Completed' },
        { number: '50K+', label: 'Active Users' },
        { number: '99.9%', label: 'Uptime' },
        { number: '24/7', label: 'Support Available' }
    ];

    const team = [
        {
            name: 'Sarah Johnson',
            role: 'CEO & Founder',
            description: 'Former ML researcher with 15+ years in NLP and AI. Passionate about making sentiment analysis accessible.'
        },
        {
            name: 'Michael Chen',
            role: 'CTO',
            description: 'Tech visionary specializing in scalable AI systems. Led engineering teams at leading tech companies.'
        },
        {
            name: 'Emily Rodriguez',
            role: 'Head of Product',
            description: 'Product strategist focused on user experience. Ensures our tools meet real-world business needs.'
        }
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900 px-4 py-12">
            <div className="max-w-6xl mx-auto">
                {/* Hero Section */}
                <header className="text-center mb-16">
                    <h1 className="text-5xl font-bold text-white mb-6">About Us</h1>
                    <p className="text-gray-300 text-lg leading-relaxed max-w-3xl mx-auto">
                        We're a team of passionate developers, data scientists, and AI enthusiasts dedicated to
                        revolutionizing how businesses understand and analyze sentiment in text data.
                    </p>
                </header>

                {/* Stats Section */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-16 bg-gray-800/20 backdrop-blur-sm border border-gray-700 rounded-2xl p-8">
                    {stats.map(stat => (
                        <StatCard key={stat.label} {...stat} />
                    ))}
                </div>

                {/* Story Section */}
                <section className="mb-16">
                    <div className="bg-gray-800/20 backdrop-blur-sm border border-gray-700 rounded-2xl p-8">
                        <h2 className="text-3xl font-bold text-white mb-6">Our Story</h2>
                        <div className="space-y-4 text-gray-300 leading-relaxed">
                            <p>
                                Founded in 2020, Sentiment Analyzer was born from a simple observation: businesses were
                                drowning in customer feedback but lacking the tools to extract meaningful insights quickly
                                and accurately.
                            </p>
                            <p>
                                Our founders, a team of MIT graduates specializing in natural language processing, set out
                                to create a solution that would democratize sentiment analysis. What started as a research
                                project quickly evolved into a powerful platform serving thousands of users worldwide.
                            </p>
                            <p>
                                Today, we continue to push the boundaries of what's possible with AI-powered text analysis,
                                helping businesses make data-driven decisions and better understand their customers.
                            </p>
                        </div>
                    </div>
                </section>

                {/* Features Section */}
                <section className="mb-16">
                    <h2 className="text-3xl font-bold text-white text-center mb-10">What Sets Us Apart</h2>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {features.map(feature => (
                            <FeatureCard key={feature.title} {...feature} />
                        ))}
                    </div>
                </section>

                {/* Team Section */}
                <section className="mb-16">
                    <h2 className="text-3xl font-bold text-white text-center mb-10">Meet Our Team</h2>
                    <div className="grid md:grid-cols-3 gap-6">
                        {team.map(member => (
                            <TeamMember key={member.name} {...member} />
                        ))}
                    </div>
                </section>

                {/* CTA Section */}
                <section className="text-center bg-gradient-to-r from-purple-500/20 to-blue-500/20 backdrop-blur-sm border border-purple-500/30 rounded-2xl p-12">
                    <h2 className="text-3xl font-bold text-white mb-4">Ready to Get Started?</h2>
                    <p className="text-gray-300 mb-8 max-w-2xl mx-auto">
                        Join thousands of businesses already using Sentiment Analyzer to transform their
                        customer feedback into actionable insights.
                    </p>
                    <div className="flex gap-4 justify-center flex-wrap">
                        <button className="bg-purple-500 hover:bg-purple-600 text-white font-medium px-8 py-3 rounded-lg transition-colors duration-200 shadow-lg hover:shadow-xl active:scale-[0.98] cursor-pointer" onClick={() => navigation('/sentiment-analysis')}>
                            Try It Free
                        </button>
                        <button className="bg-gray-800 hover:bg-gray-700 text-white font-medium px-8 py-3 rounded-lg border border-gray-600 transition-colors duration-200 active:scale-[0.98] cursor-pointer" onClick={() => navigation('/contact')}>
                            Contact Sales
                        </button>
                    </div>
                </section>
            </div>
        </div>
    );
}