import { NextResponse } from 'next/server';

export async function POST() {
  const response = NextResponse.json({ success: true }, { status: 200 });
  response.cookies.delete('gtx-auth-token');
  return response;
}
