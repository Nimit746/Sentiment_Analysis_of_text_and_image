
import React from 'react'

const Card = (props) => {
    return (
        <div className='mt-6 sm:mt-8 md:mt-10 text-white glass cursor-pointer hover:scale-105 transition-all p-4 sm:p-5 md:p-6 rounded-lg w-full'>
            <img className='w-full h-auto rounded-md' alt={props.heading}>{props.img}</img>
            <h1 className='text-lg sm:text-xl md:text-2xl font-semibold mt-3 sm:mt-4'>{props.heading}</h1>
            <p className='text-sm sm:text-base md:text-lg mt-2 sm:mt-3'>{props.data}</p>
        </div>
    )
}

export default Card