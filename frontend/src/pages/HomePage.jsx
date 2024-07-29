import React from 'react';
import Navbar from '../components/Navbar/Navbar';   // import { Navbar } from '../components/Navbar/Navbar';    // this will error as we are not use export const Navbar = () => { ... };
import Footer from '../components/Footer/Footer';
import HomePageContent from '../contents/HomePageContent';

const HomePage = () => {
    return (
        <div>
            {/* Navbar */}
            <Navbar />

            {/* Main Content */}
            <HomePageContent />

            {/* Footer */}
            <Footer />
        </div>
    );
};

export default HomePage;