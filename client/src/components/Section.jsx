import React from 'react'
import Card from './Card'

const Section = (props) => {
    return (
        <section className='mt-20'>
            <div className='flex flex-col'>
                <h1 className='text-white font-bold text-3xl'>{props.heading}</h1>
                <h1 className='text-white font-bold text-5xl mt-10'>{props.subheading}</h1>
                <p className='text-white font-semibold text-xl mt-5'>{props.para}</p>
            </div>

            <div className='flex gap-2 min-w-full'>
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
