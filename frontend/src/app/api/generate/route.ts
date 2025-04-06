import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { email, purpose } = await request.json();
    
    if (!email || !purpose) {
      return NextResponse.json(
        { message: 'Email and purpose are required' },
        { status: 400 }
      );
    }

    // Generate random credentials
    const apiKey = generateRandomString(32);
    const password = generateRandomString(16);

    // In production: Store these in your database
    // and associate with the user's email

    return NextResponse.json({
      apiKey,
      password,
      message: 'API credentials generated successfully'
    });
  } catch (error) {
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    );
  }
}

function generateRandomString(length: number): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  return Array.from({ length }, () => 
    chars.charAt(Math.floor(Math.random() * chars.length))
  ).join('');
}