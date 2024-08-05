import Image from 'react-bootstrap/Image'
import './HeroSection.css'
import Col from 'react-bootstrap/Col'
import Row from 'react-bootstrap/Row'
import Container from 'react-bootstrap/Container'
import { InlineMath } from 'react-katex'
import Nav from 'react-bootstrap/Nav';


function HeroSection(){
    var Latex = require('react-latex');
    return(
        <Container fluid className="hero-container">
            <Row>
                <Col md={6} className="text-col">
                    <h1 className='hero-large-text'>
                        Turn Images Into{' '}
                        <InlineMath math="\LaTeX"></InlineMath> {' '}
                        Markup for Free
                    </h1>
                    <h4 className='hero-small-text'>
                        Quickly generate formulas from images with just a click of a button!
                    </h4>
                    <Container fluid className="try-it-button text-center">
                    <Nav.Link href='/create'>
                        <button className="btn btn-primary">Try It</button>
                    </Nav.Link>
                    </Container>
                </Col>
                <Col md={6} className="image-col">
                    <Image className='hero-image'src="/hero_image.001.jpeg"></Image>
                </Col>
            </Row>
        </Container>
    )
}

export default HeroSection;