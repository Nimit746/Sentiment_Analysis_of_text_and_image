import React from 'react'

const Card = (props) => {
    return (
        <div className='mt-10 text-white glass cursor-pointer hover:scale-105 transition-all'>
            <img className=''>{props.img}</img>
            <h1>{props.heading}</h1>
            <p className=''>{props.data}</p>
        </div>
    )
}

export default Card
