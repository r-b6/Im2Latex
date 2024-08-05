import React from 'react';
import Container from 'react-bootstrap/Container';
import Row from 'react-bootstrap/Row';
import Col from 'react-bootstrap/Col';
import Image from 'react-bootstrap/Image';
import './TutorialComponent.css';

export default function TutorialComponent() {
    return (
        <Container className='tutorial-container'>
            <Row className='tutorial-row'>
                <Col md={12} className='text-column'>
                    <h2>How to use this tool</h2>
                    <Row className='item-row' md={12}>
                    <Col md={6}>
                    <p>Step 1: Crop your image to just its formula area, ensuring
                        there are no margins on the sides or top. 
                    </p>
                    </Col>
                    <Col md={6}>
                    <Image src='/f025417273d2f29_basic.png' fluid className='tutorial-image first' alt='Step 1'/>
                    </Col>
                    </Row>
                    <Row className='item-row' md={12}>
                    <Col md={6}>
                    <p>Step 2: Go to the create page of this website and upload
                        your image. Within a few seconds, its corresponding
                        markup will be written to the textbox on the right. 
                    </p>
                    </Col>
                    <Col md={6}>
                    <Image src='image2.png' fluid className='tutorial-image' alt='Step 2'/>
                    </Col>
                    </Row>
                    <Row md={12} className='item-row'>
                    <Col md={6}>
                    <p>Step 3: Save your text by either downloading it 
                        as a .txt file or copying it to your clipboard. 
                    </p>
                    </Col>
                    <Col md={6}>
                    <Image src='image3.png' fluid className='tutorial-image' alt='Step 3'/>
                    </Col>
                    </Row>
                </Col>
                <div>
                    <p className='footnote'>
                        *Note that the current implementation only supports 
                        images of text, like the one shown above. Handwritten 
                        formulas are <b>not</b> currently supported, although
                        this may change in future iterations. 
                    </p>
                </div>
            </Row>
        </Container>
    );
}
