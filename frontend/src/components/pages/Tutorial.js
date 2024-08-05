
import Container from 'react-bootstrap/Container';
import './Tutorial.css';
import TutorialComponent from '../TutorialComponent.js';

export default Tutorial

function Tutorial(){
    return(
        <Container fluid className='tutorial-container'>
            <TutorialComponent></TutorialComponent>
        </Container>
    )
}