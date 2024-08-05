import Image from 'react-bootstrap/Image'
import Col from 'react-bootstrap/Col'
import Row from 'react-bootstrap/Row'
import Container from 'react-bootstrap/Container'
import TextDisplayer from '../TextDisplayer'
import {useState} from 'react'
import "./Create.css";

export default Create;

function Create(){
    const [uploadedFile, setUploadedFile] = useState(null);
    const handleUpload = (event) => {
        const formData = new FormData();
        setUploadedFile(URL.createObjectURL(event.target.files[0]));
        formData.append('image', event.target.files[0])
        fetch('http://127.0.0.1:5000/create', {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/json',
            }
        }).then(response => response.json()).then(data => {
            data.error ? setTextVal(data.error) : 
            setTextVal("" + data.prediction)
        }).catch(error => {
            console.error('Error:', error);
            setTextVal('An error occurred while uploading the image');
    });
        
}
    const [textVal, setTextVal] = useState("");
    const handleTextVal = (event) => {setTextVal(event.target.value)}

    return (
        <Container fluid className='create-container'>
            <Row>
                <Col md={6} className='upload-col d-flex align-content-center justify-content-center position-relative'>
                <div> 
                    {uploadedFile ? 
                    <Image src={uploadedFile} className='uploaded-image w-80% h-auto'></Image> :
                    null
                    }
                    <input type='file' className='btn-primary' onChange={handleUpload}
                    accept='image/png image/jpg image/jpeg image/tiff image/tif'
                    ></input>
                    </div>
                    <div className="vertical-line"></div>
                </Col>
                <Col md={6} className='output-col d-flex align-content-center justify-content-center flex-column'>
                <TextDisplayer 
                includeButton={true}
                textVal={textVal}
                handleTextChange={handleTextVal}
                ></TextDisplayer>
                </Col>
            </Row>
        </Container>
    )
    };

