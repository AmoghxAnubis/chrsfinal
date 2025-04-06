"use client";

import { BrainCircuit, Linkedin, Mail, Twitter } from "lucide-react";
import Link from "next/link";
import Spline from '@splinetool/react-spline';
import { useAuth } from "@/context/auth-context";
import { useState } from "react";

interface TeamMember {
  name: string;
  role: string;
  bio: string;
  photo: string;
  social: {
    linkedin?: string;
    twitter?: string;
    email?: string;
  };
}

export default function CompanyPage() {
  const { user } = useAuth();
  const [splineLoaded, setSplineLoaded] = useState(false);

  const teamMembers: TeamMember[] = [
    {
      name: "Ayush Pal",
      role: "Frontend dev",
      bio: "Like to make things look appealing",
      photo: "/team/photo_2025-04-06_03-15-14 (1).png",
      social: {
      
        email: "palayush514@gmail.com"
      }
    },
    {
      name: "Amogh Sharma",
      role: "Backend Dev",
      bio: "Building the backbone of everything",
      photo: "/team/mine photo (1).png",
      social: {
        
        email: "amoghzach321@gmial.com"
      }
    },
    {
      name: "Aditya Veer Singh",
      role: "Flask Dev",
      bio: "I make whatever they make functional",
      photo: "/team/ballu photo (1).png",
      social: {
        
        email: "adityaveer.singh.2212@gmail.com"
      }
    }
  ];

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Background with Spline and Gradient */}
      <div className="fixed inset-0 -z-50 h-screen w-full overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-[#2e1065] via-[#4c1d95] to-[#6d28d9] opacity-90" />
        
        <div className={`absolute inset-0 transition-opacity duration-1000 ${splineLoaded ? 'opacity-100' : 'opacity-0'}`}>
          <div className="absolute inset-0 scale-[3.3] origin-center">
            <Spline 
              scene="https://prod.spline.design/Q2ySEkAE51u4pReL/scene.splinecode"
              onLoad={() => setSplineLoaded(true)}
              className="w-full h-full mix-blend-lighten"
            />
          </div>
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(139,92,246,0.5)_0%,transparent_70%)] pointer-events-none" />
        </div>
        <div className="absolute inset-0 bg-[#0f0c29]/20 backdrop-blur-[1px]" />
      </div>

      {/* Navigation */}
      <header className="fixed w-full z-50 group">
        <div className="absolute inset-x-0 top-0 h-20 bg-gradient-to-b from-[#3a1c6e]/70 to-transparent backdrop-blur-sm transition-all duration-300 group-hover:backdrop-blur-md group-hover:from-[#3a1c6e]/90" />
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <Link href="/" className="flex items-center hover:[text-shadow:_0_0_15px_rgba(159,122,234,0.9)] transition-all duration-300">
              <BrainCircuit className="h-8 w-8 text-purple-300 mr-2 drop-shadow-[0_0_8px_rgba(159,122,234,0.8)] hover:drop-shadow-[0_0_15px_rgba(159,122,234,1)] transition-all duration-300" />
              <span className="text-xl font-bold text-purple-100 [text-shadow:_0_0_10px_rgba(159,122,234,0.7)]">
                CHRS
              </span>
            </Link>

            <nav className="hidden md:flex items-center space-x-8">
              <Link href="/" className="text-purple-100 hover:text-white transition-all duration-300 [text-shadow:_0_0_8px_rgba(159,122,234,0.6)] hover:[text-shadow:_0_0_15px_rgba(255,255,255,0.8)]">
                Home
              </Link>
              <Link href="/product" className="text-purple-100 hover:text-white transition-all duration-300 [text-shadow:_0_0_8px_rgba(159,122,234,0.6)] hover:[text-shadow:_0_0_15px_rgba(255,255,255,0.8)]">
                Product
              </Link>
              <Link href="/company" className="text-purple-100 hover:text-white transition-all duration-300 [text-shadow:_0_0_8px_rgba(159,122,234,0.6)] hover:[text-shadow:_0_0_15px_rgba(255,255,255,0.8)]">
                Company
              </Link>
            </nav>

            {user ? (
              <Link 
                href="/dashboard" 
                className="text-purple-100 hover:text-white transition-all duration-300 [text-shadow:_0_0_8px_rgba(159,122,234,0.6)] hover:[text-shadow:_0_0_15px_rgba(255,255,255,0.8)]"
              >
                Dashboard
              </Link>
            ) : (
              <Link 
                href="/login" 
                className="text-purple-100 hover:text-white transition-all duration-300 [text-shadow:_0_0_8px_rgba(159,122,234,0.6)] hover:[text-shadow:_0_0_15px_rgba(255,255,255,0.8)]"
              >
                Login
              </Link>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="relative z-10 min-h-screen pt-28 pb-16 px-4">
        <div className="max-w-7xl mx-auto">
          {/* Hero Section */}
          <div className="text-center mb-16">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-4 text-white drop-shadow-lg hover:[text-shadow:_0_0_20px_rgba(255,255,255,0.3)] transition-all duration-500">
              Team Divinity
            </h1>
            <p className="text-xl text-purple-200 max-w-3xl mx-auto">
              With CHRS our goal is pioneering the future of AI-driven drug discovery with a passionate team of scientists, engineers, and researchers.
            </p>
          </div>

          {/* Team Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {teamMembers.map((member, index) => (
              <div 
                key={index}
                className="bg-[#1e1b4b]/50 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20 hover:border-purple-500/40 transition-all duration-300 hover:scale-[1.02]"
              >
                <div className="flex flex-col items-center">
                  {/* Team Member Photo */}
                  <div className="relative w-32 h-32 mb-4 rounded-full overflow-hidden border-2 border-purple-500/30 hover:border-purple-500/60 transition-all duration-300">
                    <img 
                      src={member.photo} 
                      alt={member.name}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  
                  <h3 className="text-2xl font-bold text-white mb-1 text-center">{member.name}</h3>
                  <p className="text-purple-300 mb-4 text-center">{member.role}</p>
                  <p className="text-purple-100 text-center mb-4">{member.bio}</p>
                  
                  <div className="flex space-x-4 mt-2">
                    {member.social.linkedin && (
                      <Link 
                        href={member.social.linkedin}
                        className="text-purple-300 hover:text-white transition-colors duration-200"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <Linkedin className="w-5 h-5" />
                      </Link>
                    )}
                    {member.social.twitter && (
                      <Link 
                        href={member.social.twitter}
                        className="text-purple-300 hover:text-white transition-colors duration-200"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <Twitter className="w-5 h-5" />
                      </Link>
                    )}
                    {member.social.email && (
                      <Link 
                        href={`mailto:${member.social.email}`}
                        className="text-purple-300 hover:text-white transition-colors duration-200"
                      >
                        <Mail className="w-5 h-5" />
                      </Link>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Company Mission */}
          <div className="mt-16 bg-[#1e1b4b]/70 backdrop-blur-sm rounded-xl p-8 border border-purple-500/30">
            <h2 className="text-3xl font-bold text-white mb-6 text-center">
              Our Mission
            </h2>
            <p className="text-purple-200 text-center max-w-4xl mx-auto text-lg">
              At CHRS, we're revolutionizing drug discovery by combining cutting-edge AI with deep scientific expertise. 
              Our platform accelerates the identification of promising compounds while reducing development costs, 
              bringing life-saving treatments to patients faster.
            </p>
          </div>
        </div>
      </div>
    </main>
  );
}