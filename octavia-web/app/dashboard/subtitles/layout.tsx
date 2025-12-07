
export default function SubtitlesLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col h-full">
      <main className="flex-1 overflow-y-auto p-6">
        {children}
      </main>
    </div>
  );
}
