import Container from 'react-bootstrap/Container';
import Nav from 'react-bootstrap/Nav';
import Navbar from 'react-bootstrap/Navbar';

function NavBar() {
    // const [loggedIn, setLoggedIn] = useState(false);
    // const user = {
    //     profilePicture: 'https://example.com/user-profile-picture.jpg', 
    //   };

    return (
        <Navbar expand="lg" className="bg-body-tertiary">
          <Container>
            <Navbar.Brand href="/">
            I2L <i class="fa-solid fa-cube"></i>
            </Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
              <Nav className="me-auto">
                <Nav.Link href="/tutorial">Tutorial</Nav.Link>
                <Nav.Link href="/create">Create</Nav.Link>
              </Nav>
            <Nav className="ms-auto">
            {/* {loggedIn ? (
            <Nav.Link href="/profile">
            <img 
              src={user.profilePicture} 
              alt="My Profile" 
              className="user-profile-picture"
            />
          </Nav.Link>)
            :(
            <Nav.Link href="/login">
              <button className="btn btn-primary">Log In</button>
            </Nav.Link>)
            } */}
            </Nav>
            </Navbar.Collapse>
          </Container>
        </Navbar>
      );
    }
    

export default NavBar;