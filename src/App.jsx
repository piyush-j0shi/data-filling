import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'

const API_HEADERS = {
  'Content-Type': 'application/json',
  'ngrok-skip-browser-warning': 'true',
}

async function fetchSubmissions(username) {
  try {
    const res = await fetch(
      `/api/submissions?username=${encodeURIComponent(username)}`,
      { headers: API_HEADERS }
    )
    return res.ok ? await res.json() : []
  } catch {
    return []
  }
}

async function postSubmission(username, data) {
  try {
    await fetch('/api/submissions', {
      method: 'POST',
      headers: API_HEADERS,
      body: JSON.stringify({ username, ...data }),
    })
  } catch {
    // network error — submission still updates local state
  }
}

function LoginPage({ onLogin, error }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    onLogin(username, password)
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative">
      <div className="absolute top-4 right-4">
        <DarkModeToggle />
      </div>
      <Card className="w-full max-w-sm">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">Login</CardTitle>
          <CardDescription>Enter your credentials to continue</CardDescription>
        </CardHeader>
        <CardContent>
          {error && (
            <p className="text-sm text-destructive mb-4">{error}</p>
          )}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
            <Button type="submit" className="w-full">
              Login
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}

function SubmissionDetail({ submission, index, open, onOpenChange }) {
  if (!submission) return null

  const fields = [
    { label: 'Name', value: submission.name, icon: <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg> },
    { label: 'Email', value: submission.email, icon: <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg> },
    { label: 'Phone', value: submission.phone, icon: <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg> },
    { label: 'Address', value: submission.address, icon: <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg> },
  ]

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <span className="flex h-7 w-7 items-center justify-center rounded-full bg-primary text-primary-foreground text-xs font-bold">
              {index}
            </span>
            {submission.name}
          </DialogTitle>
          <DialogDescription>Submission details</DialogDescription>
        </DialogHeader>

        <div className="space-y-3 pt-2">
          {fields.map((field) => (
            <div key={field.label} className="flex items-start gap-3 rounded-lg border bg-muted/40 p-3">
              <div className="mt-0.5 text-muted-foreground shrink-0">
                {field.icon}
              </div>
              <div className="min-w-0">
                <p className="text-xs font-medium text-muted-foreground">{field.label}</p>
                <p className="text-sm mt-0.5 break-words">{field.value}</p>
              </div>
            </div>
          ))}

          <div className="rounded-lg border bg-muted/40 p-3">
            <div className="flex items-center gap-2 mb-1.5">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              <p className="text-xs font-medium text-muted-foreground">Message</p>
            </div>
            <p className="text-sm whitespace-pre-wrap break-words">{submission.message}</p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

function SubmissionsTable({ submissions }) {
  const [selected, setSelected] = useState(null)

  if (submissions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <div className="rounded-full bg-muted p-3 mb-3">
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-muted-foreground"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>
        </div>
        <p className="text-sm font-medium text-muted-foreground">No submissions yet</p>
        <p className="text-xs text-muted-foreground/70 mt-1">Submissions will appear here once added.</p>
      </div>
    )
  }

  return (
    <>
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow className="bg-muted/50 hover:bg-muted/50">
              <TableHead className="w-12 text-center">#</TableHead>
              <TableHead>Name</TableHead>
              <TableHead>Email</TableHead>
              <TableHead>Phone</TableHead>
              <TableHead>Address</TableHead>
              <TableHead>Message</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {submissions.map((s, i) => (
              <TableRow
                key={i}
                className="even:bg-muted/30 cursor-pointer"
                onClick={() => setSelected(i)}
              >
                <TableCell className="text-center text-muted-foreground tabular-nums">{i + 1}</TableCell>
                <TableCell className="font-medium">{s.name}</TableCell>
                <TableCell>{s.email}</TableCell>
                <TableCell className="tabular-nums">{s.phone}</TableCell>
                <TableCell>{s.address}</TableCell>
                <TableCell className="max-w-[200px] truncate">{s.message}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <SubmissionDetail
        submission={selected !== null ? submissions[selected] : null}
        index={selected !== null ? selected + 1 : 0}
        open={selected !== null}
        onOpenChange={(open) => { if (!open) setSelected(null) }}
      />
    </>
  )
}

function LiveFormDataTable({ formData }) {
  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>Email</TableHead>
          <TableHead>Phone</TableHead>
          <TableHead>Address</TableHead>
          <TableHead>Message</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        <TableRow>
          <TableCell>{formData.name}</TableCell>
          <TableCell>{formData.email}</TableCell>
          <TableCell>{formData.phone}</TableCell>
          <TableCell>{formData.address}</TableCell>
          <TableCell>{formData.message}</TableCell>
        </TableRow>
      </TableBody>
    </Table>
  )
}

function FormPage({ onSubmit }) {
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
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>New Entry</CardTitle>
          <CardDescription>Fill out the fields below and submit.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="name">Name</Label>
                <Input id="name" name="name" value={formData.name} onChange={handleChange} required />
              </div>
              <div className="space-y-2">
                <Label htmlFor="email">Email</Label>
                <Input id="email" name="email" type="email" value={formData.email} onChange={handleChange} required />
              </div>
              <div className="space-y-2">
                <Label htmlFor="phone">Phone</Label>
                <Input id="phone" name="phone" value={formData.phone} onChange={handleChange} required />
              </div>
              <div className="space-y-2">
                <Label htmlFor="address">Address</Label>
                <Input id="address" name="address" value={formData.address} onChange={handleChange} required />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="message">Message</Label>
              <Textarea id="message" name="message" value={formData.message} onChange={handleChange} required />
            </div>
            <Button type="submit" className="w-full">Submit</Button>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Live Form Data</CardTitle>
        </CardHeader>
        <CardContent>
          <LiveFormDataTable formData={formData} />
        </CardContent>
      </Card>
    </div>
  )
}

function ViewSubmissions({ submissions }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle>All Submissions</CardTitle>
            <Badge variant="secondary">{submissions.length}</Badge>
          </div>
          {submissions.length > 0 && (
            <p className="text-xs text-muted-foreground">
              {submissions.length} {submissions.length === 1 ? 'record' : 'records'}
            </p>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <SubmissionsTable submissions={submissions} />
      </CardContent>
    </Card>
  )
}

function DarkModeToggle() {
  const [dark, setDark] = useState(() =>
    document.documentElement.classList.contains('dark')
  )

  useEffect(() => {
    if (dark) {
      document.documentElement.classList.add('dark')
      localStorage.setItem('theme', 'dark')
    } else {
      document.documentElement.classList.remove('dark')
      localStorage.setItem('theme', 'light')
    }
  }, [dark])

  useEffect(() => {
    const saved = localStorage.getItem('theme')
    if (saved === 'dark' || (!saved && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      setDark(true)
    }
  }, [])

  return (
    <Button variant="outline" size="icon" onClick={() => setDark(!dark)} aria-label="Toggle dark mode">
      {dark ? (
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
      ) : (
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      )}
    </Button>
  )
}

function App() {
  const [loggedIn, setLoggedIn] = useState(false)
  const [loginError, setLoginError] = useState('')
  const [submissions, setSubmissions] = useState([])
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
    setUsername('')
    setSubmissions([])
  }

  const handleFormSubmit = async (formData) => {
    setSubmissions([...submissions, formData])
    postSubmission(username, formData)
  }

  if (!loggedIn) {
    return <LoginPage onLogin={handleLogin} error={loginError} />
  }

  return (
    <div className="min-h-screen bg-muted/30">
      <div className="mx-auto max-w-5xl p-6 space-y-6">
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground">
            Logged in as <span className="font-medium text-foreground">{username}</span>
          </p>
          <div className="flex items-center gap-2">
            <DarkModeToggle />
            <Button variant="outline" size="sm" onClick={handleLogout}>
              Logout
            </Button>
          </div>
        </div>

        <Tabs defaultValue="form">
          <TabsList className="w-full">
            <TabsTrigger value="form" className="flex-1">New Entry</TabsTrigger>
            <TabsTrigger value="view" className="flex-1">
              Submissions
              {submissions.length > 0 && (
                <Badge variant="secondary" className="ml-2">{submissions.length}</Badge>
              )}
            </TabsTrigger>
          </TabsList>
          <TabsContent value="form">
            <FormPage onSubmit={handleFormSubmit} />
          </TabsContent>
          <TabsContent value="view">
            <ViewSubmissions submissions={submissions} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

export default App
