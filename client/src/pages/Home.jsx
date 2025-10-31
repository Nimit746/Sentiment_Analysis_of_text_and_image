import React from 'react'
import { useNavigate } from 'react-router-dom'
import Section from '../components/Section';
import Footer from '../components/Footer';

const Home = () => {

    const navigation = useNavigate();
    return (
        <main className='max-w-[60%] mx-auto mt-20 min-h-screen'>
            <section className="flex flex-col justify-center items-center rounded-xl gap-6 p-4 min-h-[50vh] w-full bg-google text-white">
                <h1 className='text-5xl font-semibold'>Unlock the Power of Sentiment Analysis</h1>
                <p className='max-w-[90%] text-center'>Gain deep insights into customer feedback, social media trends, and market sentiment with our advanced sentiment analysis. Understand emotions, indentify patterns, and make data-driven decisions.</p>
                <button className='cursor-pointer p-2 bg-[#7d1ad4] rounded-xl min-w-30 font-bold' onClick={() => navigation("/sentiment-analysis")}>Try Out</button>
            </section>

            <Section
                heading='How it Works'
                subheading='Analyze Sentiment in Three Simple Steps'
                para='Our sentiment analysis is designed to be intuitive and user-friendly. Follow these steps to gain valuable insights from your data.'
                data="Easily input your text data, whether it's customer reviews, social media posts or survey responses."
                data2="Our advanced algorithms analyze the text to identify sentiment, emotions, and key themes."
                data3="Explore interactive visualizations and reports to understand sentiment trends and patterns."
                card="Input Your Data"
                card2="Analyze and Process"
                card3="Visualize Insights"
            />
            <Section
                heading='Key Features'
                subheading='Empower Your Business with Sentiment Insights'
                para='Our sentiment analysis platform offers a comprehensive suite of tools to help you understand and leverage sentiment data efficiently.'
                data="Our sentiment analysis platform offers a comprehensive suite of tools to help you understand and leverage sentiment data effectively."
                data2="Monitor sentiment trends in real-time across various platforms, including social media, customer reviews, and news articles."
                data3="Segment your audience based on sentiment to tailor your messaging and improve customer engagement."
                card="Empower Your Business with Sentiment Insights"
                card2="Real-time Sentiment Tracking"
                card3="Audience Segmentation"
            />
            <section className='mt-30 flex flex-col gap-10 text-white'>
                <h1 className='text-3xl font-bold'>Visual Demo</h1>
                <video src="/public/Demo_video.mp4" alt="Demo Video" controls className='rounded-xl min-h-[60vh]'/>
            </section>
            <Footer/>
        </main>
    )
}
export default Home