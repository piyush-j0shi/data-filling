import { useState } from 'react'

async function fetchSubmissions(username) {
  const res = await fetch(`/api/submissions?username=${encodeURIComponent(username)}`)
  return res.ok ? await res.json() : []
}

async function postSubmission(username, data) {
  await fetch('/api/submissions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, ...data }),
  })
}

function LoginPage({ onLogin, error }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    onLogin(username, password)
  }

  return (
    <div style={{ maxWidth: 300, margin: '100px auto' }}>
      <h2>Login</h2>
      {error && <p style={{ color: 'red' }}>{error}</p>}
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: 10 }}>
          <label>Username</label><br />
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </div>
        <div style={{ marginBottom: 10 }}>
          <label>Password</label><br />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit">Login</button>
      </form>
    </div>
  )
}

function SubmissionsTable({ submissions }) {
  if (submissions.length === 0) {
    return <p>No submissions yet.</p>
  }

  return (
    <table border="1" cellPadding="5" style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          <th>#</th>
          <th>Name</th>
          <th>Email</th>
          <th>Phone</th>
          <th>Address</th>
          <th>Message</th>
        </tr>
      </thead>
      <tbody>
        {submissions.map((s, i) => (
          <tr key={i}>
            <td>{i + 1}</td>
            <td>{s.name}</td>
            <td>{s.email}</td>
            <td>{s.phone}</td>
            <td>{s.address}</td>
            <td>{s.message}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function LiveFormDataTable({ formData }) {
  return (
    <table border="1" cellPadding="5" style={{ width: '100%', borderCollapse: 'collapse' }}>
      <thead>
        <tr>
          <th>Name</th>
          <th>Email</th>
          <th>Phone</th>
          <th>Address</th>
          <th>Message</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>{formData.name}</td>
          <td>{formData.email}</td>
          <td>{formData.phone}</td>
          <td>{formData.address}</td>
          <td>{formData.message}</td>
        </tr>
      </tbody>
    </table>
  )
}

function FormPage({ submissions, onSubmit }) {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    address: '',
    message: '',
  })

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    onSubmit(formData)
    setFormData({ name: '', email: '', phone: '', address: '', message: '' })
  }

  return (
    <>
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: 10 }}>
          <label>Name</label><br />
          <input name="name" value={formData.name} onChange={handleChange} required />
        </div>
        <div style={{ marginBottom: 10 }}>
          <label>Email</label><br />
          <input name="email" type="email" value={formData.email} onChange={handleChange} required />
        </div>
        <div style={{ marginBottom: 10 }}>
          <label>Phone</label><br />
          <input name="phone" value={formData.phone} onChange={handleChange} required />
        </div>
        <div style={{ marginBottom: 10 }}>
          <label>Address</label><br />
          <input name="address" value={formData.address} onChange={handleChange} required />
        </div>
        <div style={{ marginBottom: 10 }}>
          <label>Message</label><br />
          <textarea name="message" value={formData.message} onChange={handleChange} required />
        </div>
        <button type="submit">Submit</button>
      </form>

      <div style={{ marginTop: 20 }}>
        <h3>Live Form Data</h3>
        <LiveFormDataTable formData={formData} />
      </div>

      {submissions.length > 0 && (
        <div style={{ marginTop: 30 }}>
          <h3>Submitted Data ({submissions.length})</h3>
          <SubmissionsTable submissions={submissions} />
        </div>
      )}
    </>
  )
}

function ViewSubmissions({ submissions }) {
  return (
    <>
      <h3>All Submissions ({submissions.length})</h3>
      <SubmissionsTable submissions={submissions} />
    </>
  )
}

function App() {
  const [loggedIn, setLoggedIn] = useState(false)
  const [loginError, setLoginError] = useState('')
  const [submissions, setSubmissions] = useState([])
  const [currentPage, setCurrentPage] = useState('form')
  const [username, setUsername] = useState('')

  const handleLogin = async (user, password) => {
    if (!user || !password) {
      setLoginError('Username and password are required')
      return
    }

    setUsername(user)
    setSubmissions(await fetchSubmissions(user))
    setLoggedIn(true)
    setLoginError('')
  }

  const handleLogout = () => {
    setLoggedIn(false)
    setCurrentPage('form')
    setUsername('')
    setSubmissions([])
  }

  const handleFormSubmit = async (formData) => {
    await postSubmission(username, formData)
    setSubmissions([...submissions, formData])
  }

  if (!loggedIn) {
    return <LoginPage onLogin={handleLogin} error={loginError} />
  }

  return (
    <div style={{ maxWidth: 500, margin: '40px auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
        <div style={{ display: 'flex', gap: 10 }}>
          <button
            onClick={() => setCurrentPage('form')}
            style={{ fontWeight: currentPage === 'form' ? 'bold' : 'normal' }}
          >
            Submit Form
          </button>
          <button
            onClick={() => setCurrentPage('view')}
            style={{ fontWeight: currentPage === 'view' ? 'bold' : 'normal' }}
          >
            View Submissions ({submissions.length})
          </button>
        </div>
        <button onClick={handleLogout}>Logout</button>
      </div>

      {currentPage === 'form' ? (
        <FormPage submissions={submissions} onSubmit={handleFormSubmit} />
      ) : (
        <ViewSubmissions submissions={submissions} />
      )}
    </div>
  )
}

export default App
