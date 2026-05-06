import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { username, password } = await request.json();

    // Use environment variables or fallback to default credentials
    const validUsername = process.env.ADMIN_USERNAME || 'admin';
    const validPassword = process.env.ADMIN_PASSWORD || 'gridtokenx';

    if (username === validUsername && password === validPassword) {
      // Create a successful response
      const response = NextResponse.json({ success: true }, { status: 200 });
      
      const isProduction = process.env.NODE_ENV === 'production';
      response.cookies.set({
        name: 'gtx-auth-token',
        value: 'authenticated',
        httpOnly: true,
        secure: isProduction,
        sameSite: 'lax',
        path: '/',
        maxAge: 60 * 60 * 24 * 7 // 1 week
      });
      
      return response;
    }

    return NextResponse.json({ error: 'Invalid username or password' }, { status: 401 });
  } catch (error) {
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
