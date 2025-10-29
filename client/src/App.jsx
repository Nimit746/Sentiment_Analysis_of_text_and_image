import React from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import SentimentAnalysis from './pages/SentimentAnalysis'
import About from './pages/About'
import Contact from './pages/Contact'
import Layout from './pages/Layout'
import Signup from './pages/Signup' 
import NotFound from './pages/NotFound'

const App = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path='/' element={<Layout />} >
          <Route index element={<Home />} />
          <Route path='sentiment-analysis' element={<SentimentAnalysis />} />
          <Route path='about' element={<About />} />
          <Route path='contact' element={<Contact />}/>
          {/* <Route path='signup' element={<Signup />} /> */}
          <Route path='*' element={<NotFound />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
