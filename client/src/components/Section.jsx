import React from 'react'
import Card from './Card'

const Section = (props) => {
    return (
        <section className='mt-12 sm:mt-16 md:mt-20 px-4 sm:px-6 md:px-8'>
            <div className='flex flex-col'>
                <h1 className='text-white font-bold text-xl sm:text-2xl md:text-3xl'>{props.heading}</h1>
                <h1 className='text-white font-bold text-3xl sm:text-4xl md:text-5xl mt-6 sm:mt-8 md:mt-10'>{props.subheading}</h1>
                <p className='text-white font-semibold text-base sm:text-lg md:text-xl mt-3 sm:mt-4 md:mt-5 max-w-4xl'>{props.para}</p>
            </div>

            <div className='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5 md:gap-6 w-full'>
                <Card
                    // img=''
                    heading={props.card}
                    data={props.data}
                />
                <Card
                    // img=''
                    heading={props.card2}
                    data={props.data2}
                />
                <Card
                    // img=''
                    heading={props.card3}
                    data={props.data3}
                />
            </div>
        </section>
    )
}

export default Section