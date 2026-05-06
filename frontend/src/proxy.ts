import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function proxy(request: NextRequest) {
  const token = request.cookies.get('gtx-auth-token')?.value;
  const { pathname } = request.nextUrl;

  // Check if it's a public path that doesn't require authentication
  if (
    pathname === '/login' ||
    pathname.startsWith('/api/auth') ||
    pathname.startsWith('/_next') ||
    pathname === '/favicon.ico' ||
    pathname.includes('.') // basic check for assets
  ) {
    // If authenticated and trying to visit login, redirect to dashboard
    if (pathname === '/login' && token === 'authenticated') {
      return NextResponse.redirect(new URL('/dashboard', request.url));
    }
    return NextResponse.next();
  }

  // If no token exists, redirect to login
  if (!token || token !== 'authenticated') {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  // If user is authenticated and visits root, redirect to dashboard
  if (pathname === '/') {
    return NextResponse.redirect(new URL('/dashboard', request.url));
  }

  return NextResponse.next();
}

export const config = {
  // Run middleware on all routes except static assets and API auth routes
  matcher: ['/((?!api/auth|_next/static|_next/image|favicon.ico).*)'],
};
