import { createFileRoute } from "@tanstack/react-router";
import { Phonolab } from "@/components/lab/Phonolab";

export const Route = createFileRoute("/")({ component: Home });

function Home() {
  return <Phonolab />;
}
